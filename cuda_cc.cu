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
//Map delta index to cardinal direction in reverse
__constant__ const int bdr[] = {1 << 3, 1 << 1, 1 << 0, 1 << 2};

__device__ int propagate_min(char centerChar, char neighChar, int centerGID, int neighGID) {

    int mask = -(int)(centerChar == neighChar); //Mask cases where chars are different
    int diff = centerGID - neighGID;
    int sign = diff >> 31;
    int minCandidate = centerGID - (diff & ~sign);

    return (minCandidate & mask) | (centerGID & ~mask);

}

__device__ void propagate(  int lxc, int lyc, int gxc, int gyc, int width, int height, const char* __restrict__ mat, int groupsChunk[ChunkSize * ChunkSize],
                            bool* __restrict__ blockStable, const int* __restrict__ groups, bool dirtyNeighbour, int mp, int gp) {

    if (gxc >= width || gyc >= height) return; //Bounds check

    #pragma unroll
    for (int i = 0; i < 4; i++) { //Loop over 4 neighbours
        int nlxc = lxc + dx[i]; int ngxc = gxc + dx[i];
        int nlyc = lyc + dy[i]; int ngyc = gyc + dy[i];

        //ToDo: maybe we can prevent the warp divergence caused here?
        if (!dirtyNeighbour && (nlxc >= ChunkSize || nlyc >= ChunkSize || nlxc < 0 || nlyc < 0)) continue;   //Bounds check (local)
        if (ngxc >= width || ngyc >= height || ngxc < 0 || ngyc < 0) continue;          //Bounds check (global)

        int neighGID = dirtyNeighbour ? groups[ngyc * gp + ngxc] : groupsChunk[nlyc * ChunkSize + nlxc];

        int newGID = propagate_min(mat[gyc * mp + gxc], mat[ngyc * mp + ngxc], groupsChunk[lyc * ChunkSize + lxc], neighGID);
        if (newGID < groupsChunk[lyc * ChunkSize + lxc]) {
            groupsChunk[lyc * ChunkSize + lxc] = newGID;
            *blockStable = false;
        }

    }

}

//Check if there's a dirty neighbouring chunk. If so lower the corresponding directional dirty flag on that chunk
__device__ void serialCheckDirty(int ll, bool* __restrict__ dirtyNeighbour, ChunkStatus* __restrict__ status_matrix, dim3 numBlocks, int* __restrict__ dirtyBlocks, int sp) {
    __threadfence(); //Prevent rare inconsistencies when undirtying a chunk as its still being written to
    for (int i = 0; i < 4; i++) {
        int nx = blockIdx.x + dx[i];    //New x value
        int ny = blockIdx.y + dy[i];    //New y value
        if (ny >= numBlocks.y || ny < 0 || nx >= numBlocks.x || nx < 0) continue; //Boundary check
        int index = ny * sp + nx;

        int mask = ~bdr[i];

        ChunkStatus old_status = (ChunkStatus)atomicAnd((int*)&status_matrix[index], mask);
        bool was_dirty = (old_status & bdr[i]) != 0;

        if (was_dirty) {
            *dirtyNeighbour = true;
            if ((old_status & mask) == 0) { //If the last dirty bit has been cleared by this operation decrement dirtyBlocks
                atomicAdd(dirtyBlocks, -1);
            }
        }
    }
}

//ToDo: probs there's a way to get block count without passing it as argument
__global__ void cuda_cc(int* groups, const char* __restrict__ mat, int width, int height, ChunkStatus* __restrict__ status_matrix, dim3 numBlocks, int* __restrict__ dirtyBlocks, size_t gp, size_t mp, size_t sp) {

    //Each thread will handle two cells each (hence the doubled indexes). In the memory management part we split the 32x32 chunk into two 16x32 sections.
    //In the iterative algorithm part instead we split the 32x32 chunk into a chessboard pattern of alternating cells so that we can avoid race dontions.

    int blockStartX = blockIdx.x * ChunkSize;           // Each block covers 32 columns
    int blockStartY = blockIdx.y * ChunkSize;           // and 32 rows

    int lx = threadIdx.x; 	                            //Local x index
    int ly = threadIdx.y;                               //Local y index
    int ll = ly * ChunkSize + lx;                       //Local linearized index
    int lx1 = lx;                                       //Second local x index
    int ly1 = ly + (ChunkSize/2);                       //Second local y index
    int ll1 = ly1 * ChunkSize + lx1;                    //Second local linearized index
    
    int gx = blockStartX + lx; 	                        //Global x index
    int gy = blockStartY + ly;                          //Global y index
    int gx1 = blockStartX + lx1; 	                    //Second Global x index
    int gy1 = blockStartY + ly1;                        //Second Global y index

    __shared__ int groupsChunk[ChunkSize * ChunkSize];  //Shared memory for groups of the local chunk
    __shared__ bool blockStable;                        //Is the chunk in a stable configuration?
    __shared__ bool dirtyNeighbour;                     //Are we yet to account for changes in a neighbouring chunk?
    __shared__ bool dirtyBlock;                         //Has this chunk been changed?

    bool validGlobal = true;                            //Is the first set of coordinates globally valid?
    bool validGlobal1 = true;                           //Is the second set of coordinates globally valid?

    int glg = gy * (gp / sizeof(int)) + gx;             //Global linearized index accounting for groups pitch
    int glg1 = gy1 * (gp / sizeof(int)) + gx1;          //Second global linearized index accounting for groups pitch

    //Chess pattern (vertical)
    int lyc = ly * 2 + (lx % 2);                        //Local chess y
    int gyc = blockStartY + lyc;                        //Global chess y
    int lyc1 = ly * 2 + ((lx + 1) % 2);
    int gyc1 = blockStartY + lyc1;

    //Initialize flags
    if (ll == 0) {
        blockStable = true;
        dirtyNeighbour = true;
        dirtyBlock = false;
    }

    //Bounds check
    //if (gx >= width || gy >= height) return;
    validGlobal = !(gx >= width || gy >= height);
    validGlobal1 = !(gx1 >= width || gy1 >= height);

    int big = width * height + 100;

    //Init shared memory groups
    groupsChunk[ll] = validGlobal ? groups[glg] : big;
    groupsChunk[ll1] = validGlobal1 ? groups[glg1] : big;


    __syncthreads(); //Await end of initialization

    /////////////////////// End of initialization ///////////////////////
    
    do {

        if (ll == 0) {
            blockStable = true;
            dirtyNeighbour = false;
            //ToDo: only check this every so often?
            serialCheckDirty(ll, &dirtyNeighbour, status_matrix, numBlocks, dirtyBlocks, sp / sizeof(ChunkStatus));
        }
        __syncthreads();

        propagate(lx, lyc, gx, gyc, width, height, mat, groupsChunk, &blockStable, groups, dirtyNeighbour, mp / sizeof(char), gp / sizeof(int));
        __syncthreads();
        propagate(lx, lyc1, gx, gyc1, width, height, mat, groupsChunk, &blockStable, groups, dirtyNeighbour, mp / sizeof(char), gp / sizeof(int));

        //__syncthreads(); //Sync all at the end of an iteration
        if (!blockStable) dirtyBlock = true;
        __syncthreads();
    } while (!blockStable);
    
    __threadfence();
    
    if (dirtyBlock) {
        //Race conditions shoulnd't be a concern here
        if (validGlobal) groups[glg] = groupsChunk[ll];   //Copy stable chunk to global
        if (validGlobal1) groups[glg1] = groupsChunk[ll1];

        if (ll == 0) {
            // Calculate valid neighbors. If a neighbour in one direction doesn't exist the dirty flag shoulnd't be raised because nothing
            // would be able to then lower it again
            int flags = WAITING;
            if (blockIdx.y > 0)             flags |= DIRTY_NORTH;
            if (blockIdx.x < numBlocks.x-1) flags |= DIRTY_EAST;
            if (blockIdx.y < numBlocks.y-1) flags |= DIRTY_SOUTH;
            if (blockIdx.x > 0)             flags |= DIRTY_WEST;
            //Atomically check if chunk was already dirty and make it dirty
            ChunkStatus old_status = (ChunkStatus)atomicOr( (int*)&status_matrix[blockIdx.y * (sp / sizeof(ChunkStatus)) + blockIdx.x], flags);
            if (old_status == 0) atomicAdd(dirtyBlocks, 1); // Only increment if previously clean
        }
    }
    
}

GroupMatrix cuda_cc(const CharMatrix* __restrict__ mat) {

    dim3 numBlocks( (mat->width + ChunkSize - 1) / ChunkSize, (mat->height + ChunkSize - 1) / ChunkSize );
    dim3 numThreads(ChunkSize, ChunkSize/2);

    //Initialize and allocate device memory for groups
    int* d_groups;
    size_t groups_pitch;
    GroupMatrix h_groups = initGroupsUnique(mat->width, mat->height);
    //HANDLE_ERROR(cudaMalloc((void**)&d_groups, mat->height * mat->width * sizeof(int)));
    HANDLE_ERROR(cudaMallocPitch(&d_groups, &groups_pitch, mat->width * sizeof(int), mat->height));
    //HANDLE_ERROR(cudaMemcpy(d_groups, (void*)(h_groups.groups), h_groups.width * h_groups.height * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy2D(d_groups, groups_pitch, h_groups.groups, mat->width * sizeof(int), mat->width * sizeof(int), mat->height, cudaMemcpyHostToDevice));

    //Initialize and allocate device memory for character matrix
    char* d_mat;
    size_t mat_pitch;
    //HANDLE_ERROR(cudaMalloc((void**)&d_mat, mat->height * mat->width * sizeof(char)));
    HANDLE_ERROR(cudaMallocPitch(&d_mat, &mat_pitch, mat->width * sizeof(char), mat->height));
    //Copy char matrix to device memory
    //HANDLE_ERROR(cudaMemcpy(d_mat, (void*)mat->matrix, mat->width * mat->height * sizeof(char), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy2D(d_mat, mat_pitch, mat->matrix, mat->width * sizeof(char), mat->width * sizeof(char), mat->height, cudaMemcpyHostToDevice));

    //Initialize status matrix
    int statusSize = sizeof(ChunkStatus) * numBlocks.x * numBlocks.y;
    enum ChunkStatus* h_status_matrix;
    h_status_matrix = (ChunkStatus*)malloc(statusSize);
    // Initialize status matrix with by forcing a first inter-block communication before anything else happens
    for (int x = 0; x < numBlocks.x; x++) {
        for (int y = 0; y < numBlocks.y; y++) {
            int status = WAITING;
            if (y < numBlocks.y - 1) status |= DIRTY_SOUTH;
            if (x < numBlocks.x - 1) status |= DIRTY_EAST;
            if (y > 0)               status |= DIRTY_NORTH;
            if (x > 0)               status |= DIRTY_WEST;
            h_status_matrix[y * numBlocks.x + x] = (ChunkStatus)status;
        }
    }
    enum ChunkStatus* d_status_matrix;
    //HANDLE_ERROR(cudaMalloc((void**)&d_status_matrix, statusSize));
    //HANDLE_ERROR(cudaMemcpy(d_status_matrix, h_status_matrix, statusSize, cudaMemcpyHostToDevice));
    size_t status_pitch;
    HANDLE_ERROR(cudaMallocPitch(&d_status_matrix, &status_pitch, numBlocks.x * sizeof(ChunkStatus), numBlocks.y));
    HANDLE_ERROR(cudaMemcpy2D(d_status_matrix, status_pitch, h_status_matrix, numBlocks.x * sizeof(ChunkStatus), numBlocks.x * sizeof(ChunkStatus), numBlocks.y, cudaMemcpyHostToDevice));

    //Dirty count
    int* d_dirty;
    int* h_dirty;
    cudaHostAlloc(&h_dirty, sizeof(int), 0);
    //cudaHostGetDevicePointer(&d_dirty, h_dirty, 0); Using mapped memory worsens performance considerably
    *h_dirty = numBlocks.x * numBlocks.y;
    HANDLE_ERROR(cudaMalloc((void**)&d_dirty, sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(d_dirty, h_dirty, sizeof(int), cudaMemcpyHostToDevice));

    int iters = 0;
    bool err = false;

    //Loop until stable
    while (*h_dirty > 0 && !err) {

        //printf("Dirty blocks: %d\n", *h_dirty);
        
        cuda_cc<<<numBlocks, numThreads>>>(d_groups, d_mat, mat->width, mat->height, d_status_matrix, numBlocks, d_dirty, groups_pitch, mat_pitch, status_pitch);

        HANDLE_ERROR(cudaMemcpy(h_dirty, d_dirty, sizeof(int), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        
        checkCUDAError("call of cuda_cc kernel");

        iters++;
        if (iters > numBlocks.x * numBlocks.y * 100) { //Cap of 100 kernel iterations per block
            printf("Something went wrong! Quitting and logging solution\n");
            err = true;
        }

    }

    //Copy group matrix back to host
    //HANDLE_ERROR(cudaMemcpy(h_groups.groups, (void*)d_groups, mat->width * mat->height * sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy2D(h_groups.groups, mat->width * sizeof(int), d_groups, groups_pitch, mat->width * sizeof(int), mat->height, cudaMemcpyDeviceToHost));

    if (err) {
        //Dump groups and dirty matrix to a file
        GroupMatrix dirtyMatrix;
        dirtyMatrix.width = numBlocks.x; dirtyMatrix.height = numBlocks.y;
        dirtyMatrix.groups = (int*)malloc(numBlocks.x * numBlocks.y * sizeof(int));
        //HANDLE_ERROR(cudaMemcpy(dirtyMatrix.groups, (void*)d_status_matrix, dirtyMatrix.width * dirtyMatrix.height * sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy2D(dirtyMatrix.groups, numBlocks.x * sizeof(int), &d_status_matrix, status_pitch, numBlocks.x * sizeof(int), numBlocks.y, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize(); //ToDo: check if needed
        saveGroupMatrixToFile(&h_groups, "Outputs/Errors/err_groups.txt");
        saveGroupMatrixToFile(&dirtyMatrix, "Outputs/Errors/err_statuses.txt");
    }

    //Free device memory
    HANDLE_ERROR(cudaFree(d_groups));
    HANDLE_ERROR(cudaFree(d_mat));

    cudaDeviceSynchronize(); //ToDo: check if needed

    return h_groups;

}