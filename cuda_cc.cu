#include "char_matrix.h"

#define ChunkSize 32    //Has to be divisible by 2

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "ERRORE CUDA: >%s<: >%s<. Eseguo: EXIT\n", msg, cudaGetErrorString(err) );
        exit(-1);
    }
}

enum ChunkStatus {
    WAITING = 0,
    DIRTY_NORTH = 1 << 0,
    DIRTY_EAST = 1 << 1,
    DIRTY_SOUTH = 1 << 2,
    DIRTY_WEST = 1 << 3
};

//Neighbour deltas
__constant__ const int dx[] = {1, -1, 0, 0};
__constant__ const int dy[] = {0, 0, 1, -1};
//Map delta index to cardinal direction (plus reversed direction)
//const int bd[] = {1 << 1, 1 << 3, 1 << 0, 1 << 2};
__constant__ const int bdr[] = {1 << 3, 1 << 1, 1 << 2, 1 << 0};

__device__ void propagate(int lxc, int lyc, int gxc, int gyc, int width, int height, char* mat, int groupsChunk[ChunkSize * ChunkSize], bool* blockStable) {

    if (lxc >= ChunkSize) lxc -= ChunkSize; //Faster than modulo even though it diverges?
    if (gxc >= width || gyc >= height) return; //Bounds check

    for (int i = 0; i < 4; i++) { //Loop over 4 neighbours
        int nlxc = lxc + dx[i]; int ngxc = gxc + dx[i];
        int nlyc = lyc + dy[i]; int ngyc = gyc + dy[i];
        if (nlxc >= ChunkSize || nlyc >= ChunkSize || nlxc < 0 || nlyc < 0) continue;   //Bounds check (local)
        if (ngxc >= width || ngyc >= height || ngxc < 0 || ngyc < 0) continue;          //Bounds check (global)
        bool override = (mat[gyc * width + gxc] == mat[ngyc * width + ngxc]) && groupsChunk[lyc * ChunkSize + lxc] < groupsChunk[nlyc * ChunkSize + nlxc];
        if (override) {
            groupsChunk[lyc * ChunkSize + lxc] = groupsChunk[nlyc * ChunkSize + nlxc];
            blockStable = false;
        }
    }

}

//Propagate but can access global groups and propagate from that to the local groupsChunk
__device__ void globally_propagate(int lxc, int lyc, int gxc, int gyc, int width, int height, char* mat, int groupsChunk[ChunkSize * ChunkSize], bool* blockStable, int* groups) {

    if (lxc >= ChunkSize) lxc -= ChunkSize; //Faster than modulo even though it diverges?
    if (gxc >= width || gyc >= height) return; //Bounds check

    for (int i = 0; i < 4; i++) { //Loop over 4 neighbours
        int ngxc = gxc + dx[i];
        int ngyc = gyc + dy[i];
        if (ngxc >= width || ngyc >= height || ngxc < 0 || ngyc < 0) continue;          //Bounds check (global)
        bool override = (mat[gyc * width + gxc] == mat[ngyc * width + ngxc]) && groupsChunk[lyc * ChunkSize + lxc] < groups[ngyc * width + ngxc];
        if (override) {
            groupsChunk[lyc * ChunkSize + lxc] = groups[ngyc * width + ngxc];
            blockStable = false;
        }
    }

}

__device__ void serialCheckDirty(int ll, bool* dirtyNeighbour, ChunkStatus* status_matrix, dim3 numBlocks, int* dirtyBlocks) {
    //Thread 0 checks for dirty neighbouring blocks
    if (ll == 0) {
        for (int i = 0; i < 4; i++) {
            int nx = blockIdx.x + dx[i];    //New x value
            int ny = blockIdx.y + dy[i];    //New y value
            if (ny >= numBlocks.y || ny < 0 || nx >= numBlocks.x || nx < 0) continue; //Boundary check
            int index = ny * numBlocks.x + nx;

            int mask = ~bdr[i];

            ChunkStatus old_status = (ChunkStatus)atomicAnd((int*)&status_matrix[index], mask);
            bool was_dirty = (old_status & bdr[i]) != 0;

            if (was_dirty) {
                *dirtyNeighbour = true;
                if ((old_status & mask) == 0) { //If the last dirty bit has been cleared by this operation decrement dirtyBlocks
                    atomicAdd(dirtyBlocks, -1);
                }
                break;
            }
        }
    }
    __syncthreads();
}

//ToDo: probs there's a way to get block count without passing it as argument
__global__ void cuda_cc(int* groups, char* mat, int width, int height, ChunkStatus* status_matrix, dim3 numBlocks, int* dirtyBlocks, int* busyBlocks) {

    //Each thread will handle two cells each (hence the doubled indexes). In the memory management part we split the 32x32 chunk into two 16x32 sections.
    //In the iterative algorithm part instead we split the 32x32 chunk into a chessboard pattern of alternating cells so that we can avoid race dontions.

    int blockStartX = blockIdx.x * ChunkSize;           // Each block covers 32 columns
    int blockStartY = blockIdx.y * ChunkSize;           // and 32 rows

    int lx = threadIdx.x; 	                            //Local x index
    int ly = threadIdx.y;                               //Local y index
    int ll = ly * (ChunkSize/2) + lx;                   //Local linearized index
    int lx1 = threadIdx.x + (ChunkSize/2);              //Second local x index
    int ly1 = ly;                                       //Second local y index
    int ll1 = ll + (ChunkSize/2);                       //Second local linearized index
    
    int gx = blockStartX + lx; 	                        //Global x index
    int gy = blockStartY + ly;                          //Global y index
    int gl = gy * width + gx;                           //Global linearized index
    int gx1 = blockStartX + lx1; 	                    //Second Global x index
    int gy1 = blockStartY + ly1;                        //Second Global y index
    int gl1 = gy1 * width + gx1;                        //Second Global linearized index

    __shared__ int groupsChunk[ChunkSize * ChunkSize];  //Shared memory for groups of the local chunk
    __shared__ bool blockStable;                        //Is the chunk in a stable configuration?
    __shared__ bool dirtyNeighbour;                     //Are we yet to account for changes in a neighbouring chunk?
    __shared__ bool dirtyBlock;                         //Has this chunk been changed?

    bool threadActive = true;                           //Is the thread disabled due to out of bounds coordinates?

    //Initialize flags
    if (ll == 0) {
        blockStable = true;
        dirtyNeighbour = true;
        dirtyBlock = false;
    }

    //Bounds check
    //if (gx >= width || gy >= height) return;
    threadActive = !(gx >= width || gy >= height);

    if (threadActive) {

        //Init groups
        groupsChunk[ll] = gl;
        groupsChunk[ll1] = gl1;

    }

    __syncthreads(); //Await end of initialization

    /////////////////////// End of initialization ///////////////////////

    while (true) {

        printf("dirty: %d\tbusy: %d\n", *dirtyBlocks, *busyBlocks);
    
        do {

            //ToDo: only check this every so often?
            if (!dirtyNeighbour) {
                serialCheckDirty(ll, &dirtyNeighbour, status_matrix, numBlocks, dirtyBlocks);
            }


            blockStable = true;

            //Chess pattern
            int lxc = lx * 2 + (ly % 2);    //Local chess x
            int gxc = gx + (lxc - lx);      //Global chess x

            if (!dirtyNeighbour) {
                if (threadActive) propagate(lxc, ly, gxc, gy, width, height, mat, groupsChunk, &blockStable);
                __syncthreads();
                if (threadActive) propagate(lxc + 1, ly, gxc + 1, gy, width, height, mat, groupsChunk, &blockStable);
            } else {
                if (threadActive) globally_propagate(lxc, ly, gxc, gy, width, height, mat, groupsChunk, &blockStable, groups);
                __syncthreads();
                if (threadActive) globally_propagate(lxc + 1, ly, gxc + 1, gy, width, height, mat, groupsChunk, &blockStable, groups);
                dirtyNeighbour = false;
            }

            if (!blockStable) dirtyBlock = true;

            __syncthreads(); //Sync all at the end of an iteration
        } while (!blockStable);
        
        if (dirtyBlock) {
            //Race conditions shoulnd't be a concern here
            if (threadActive) {
                groups[gl] = groupsChunk[ll];   //Copy stable chunk to global
                groups[gl1] = groupsChunk[ll1];
            }
            __syncthreads();
            if (ll == 0) {
                //Atomically check if chunk was already dirty and make it dirty
                ChunkStatus old_status = (ChunkStatus)atomicOr( (int*)&status_matrix[blockIdx.y * numBlocks.x + blockIdx.x],
                                                                DIRTY_NORTH | DIRTY_EAST | DIRTY_SOUTH | DIRTY_WEST);
                if (old_status == 0) atomicAdd(dirtyBlocks, 1); // Only increment if previously clean
            }
        }

        /////////////////////// End of main propagation loop ///////////////////////

        __syncthreads(); //Wait for other threads in the block to be done
        if (ll == 0) atomicAdd(busyBlocks, -1); //Notify that the block is no longer busy

        //Busy wait for dirty neighbours or full stabilization
        while (true) {

            //If a dirty neighbour is detected the block goes back into action
            serialCheckDirty(ll, &dirtyNeighbour, status_matrix, numBlocks, dirtyBlocks); //Note that the check includes a synchthreads at the end
            if (dirtyNeighbour) {
                if (ll == 0) atomicAdd(busyBlocks, 1);
                break;
            }
            
            //Tarmination of the algorithm if all blocks are done working and none are dirty
            if (*dirtyBlocks == 0 && *busyBlocks == 0) return;

        }

    }
    
}

GroupMatrix cuda_cc(CharMatrix* mat) {

    dim3 numBlocks( (mat->width + ChunkSize - 1) / ChunkSize, (mat->height + ChunkSize - 1) / ChunkSize );
    dim3 numThreads(ChunkSize/2, ChunkSize);

    //Initialize and allocate device memory for groups
    int* d_groups;
    HANDLE_ERROR(cudaMalloc((void**)&d_groups, mat->height * mat->width * sizeof(int)));

    //Initialize and allocate device memory for character matrix
    char* d_mat;
    HANDLE_ERROR(cudaMalloc((void**)&d_mat, mat->height * mat->width * sizeof(char)));

    //Copy char matrix to device memory
    HANDLE_ERROR(cudaMemcpy(d_mat, (void*)mat->matrix, mat->width * mat->height * sizeof(char), cudaMemcpyHostToDevice));

    //Initialize status matrix
    int statusSize = sizeof(ChunkStatus) * numBlocks.x * numBlocks.y;
    enum ChunkStatus* h_status_matrix;
    h_status_matrix = (ChunkStatus*)malloc(statusSize);
    // Initialize status matrix with by forcing a first inter-block communication before anything else happens
    // We only set south and east because of group IDs being bigger only in those direction in the initial configuration
    for (int x = 0; x < numBlocks.x; x++) {
        for (int y = 0; y < numBlocks.y; y++) {
            int status = WAITING;
            if (y < numBlocks.y - 1) status |= DIRTY_SOUTH;  // Valid south neighbor
            if (x < numBlocks.x - 1) status |= DIRTY_EAST;   // Valid east neighbor
            h_status_matrix[y * numBlocks.x + x] = (ChunkStatus)status;
        }
    }
    enum ChunkStatus* d_status_matrix;
    HANDLE_ERROR(cudaMalloc((void**)&d_status_matrix, statusSize));
    HANDLE_ERROR(cudaMemcpy(d_status_matrix, h_status_matrix, statusSize, cudaMemcpyHostToDevice));

    //Busy and dirty count
    int* d_busy;
    int h_busy = numBlocks.x * numBlocks.y;
    HANDLE_ERROR(cudaMalloc((void**)&d_busy, sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(d_busy, &h_busy, sizeof(int), cudaMemcpyHostToDevice));
    int* d_dirty;
    int h_dirty = numBlocks.x * numBlocks.y - 1; //Corner block isn't dirty to any neighbours at the start
    HANDLE_ERROR(cudaMalloc((void**)&d_dirty, sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(d_dirty, &h_dirty, sizeof(int), cudaMemcpyHostToDevice));

    cuda_cc<<<numBlocks, numThreads>>>(d_groups, d_mat, mat->width, mat->height, d_status_matrix, numBlocks, d_dirty, d_busy);
    cudaDeviceSynchronize();

    printf("Done computing on device\n");

    checkCUDAError("call of cuda_cc kernel");
    
    //Copy group matrix back to host
    GroupMatrix h_groups = simpleInitGroups(mat->width, mat->height);
    HANDLE_ERROR(cudaMemcpy(h_groups.groups, (void*)d_groups, mat->width * mat->height * sizeof(int), cudaMemcpyDeviceToHost));

    //Free device memory
    HANDLE_ERROR(cudaFree(d_groups));
    HANDLE_ERROR(cudaFree(d_mat));


    return h_groups;

}