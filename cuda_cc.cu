#include "char_matrix.h"
#include <cooperative_groups.h>

#define ChunkSize 32    //Has to be divisible by 2

//using namespace cooperative_groups;

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
__constant__ const int bdr[] = {DIRTY_WEST, DIRTY_EAST, DIRTY_NORTH, DIRTY_SOUTH};

//If chars are the same return the minimum between the two given IDs. Otherwise always return the centerID
__forceinline__ __device__ int propagate_min(char centerChar, char neighChar, int centerGID, int neighGID) {

    int mask = -(int)(centerChar == neighChar); //Mask cases where chars are different
    int diff = centerGID - neighGID;
    int sign = diff >> 31;
    int minCandidate = centerGID - (diff & ~sign);

    return (minCandidate & mask) | (centerGID & ~mask);

}

//Propagation step
__forceinline__ __device__ void propagate(  int lxc, int lyc, int gxc, int gyc, int width, int height, const char* __restrict__ mat, int groupsChunk[ChunkSize * ChunkSize],
                            bool* __restrict__ blockStable, const int* __restrict__ groups, bool dirtyNeighbour, int mp, int gp, const char* __restrict__ globalMat) {

    if (gxc >= width || gyc >= height) return; //Bounds check

    #pragma unroll
    for (int i = 0; i < 4; i++) { //Loop over 4 neighbours
        int nlxc = lxc + dx[i]; int ngxc = gxc + dx[i]; //Neighbour local and global x
        int nlyc = lyc + dy[i]; int ngyc = gyc + dy[i]; //Neighbour local and global y

        bool outOfChunkBounds = nlxc >= ChunkSize || nlyc >= ChunkSize || nlxc < 0 || nlyc < 0;

        if (!dirtyNeighbour && outOfChunkBounds) continue;   //Bounds check (local)
        if (ngxc >= width || ngyc >= height || ngxc < 0 || ngyc < 0) continue;          //Bounds check (global)

        //If a dirty neighbour has been acknowledged read from global memory when trying to reach across the chunk borders
        int neighGID = dirtyNeighbour && outOfChunkBounds ? groups[ngyc * gp + ngxc] : groupsChunk[nlyc * ChunkSize + nlxc];
        char neighVal = outOfChunkBounds ? globalMat[ngyc * mp + ngxc] : mat[nlyc * ChunkSize + nlxc];

        int newGID = propagate_min(mat[lyc * ChunkSize + lxc], neighVal, groupsChunk[lyc * ChunkSize + lxc], neighGID);
        if (newGID < groupsChunk[lyc * ChunkSize + lxc]) {
            groupsChunk[lyc * ChunkSize + lxc] = newGID;
            *blockStable = false;
        }

    }

}

//Check if there's a dirty neighbouring chunk. If so lower the corresponding directional dirty flag on that chunk
__forceinline__  __device__ void checkDirty(int ll, bool* __restrict__ dirtyNeighbour, ChunkStatus* __restrict__ status_matrix, dim3 numChunks, int* __restrict__ dirtyBlocks, int sp, dim3 chunk) {
    __threadfence(); //Prevent rare inconsistencies
    
    if (ll >= 4) return; //Only first 4 threads are needed, one per direction

    int nx = chunk.x + dx[ll];    //Neighbour x value
    int ny = chunk.y + dy[ll];    //Neighbour y value
    if (ny >= numChunks.y || ny < 0 || nx >= numChunks.x || nx < 0) return; //Boundary check
    int index = ny * sp + nx;

    int mask = ~bdr[ll];

    //Acknowledge the dirty chunk and lower its flag
    ChunkStatus old_status = (ChunkStatus)atomicAnd((int*)&status_matrix[index], mask);
    bool was_dirty = (old_status & bdr[ll]) != 0;

    if (was_dirty) {
        *dirtyNeighbour = true;
        if ((old_status & mask) == 0) { //If the last dirty bit has been cleared by this operation decrement dirtyBlocks
            atomicAdd(dirtyBlocks, -1);
        }
    }
}


__global__ void cc_kernel(int* groups, const char* __restrict__ mat, int width, int height, ChunkStatus* __restrict__ status_matrix, dim3 numChunks, int* __restrict__ dirtyBlocks, size_t gpe, size_t mpe, size_t spe) {

    //Grid-stride loop
    for (int chunkIndex = blockIdx.y * gridDim.x + blockIdx.x; chunkIndex < numChunks.x * numChunks.y; chunkIndex += gridDim.y * gridDim.x) {
    
        //Each thread will handle two cells each. In the memory management part each thread moves two adjacent cells (treating them as an int2).
        //In the iterative algorithm part instead we split the 32x32 chunk into a chessboard pattern of alternating cells so that we can avoid race dontions.

        dim3 chunk(chunkIndex % numChunks.x, chunkIndex / numChunks.x);

        int chunkStartX = chunk.x * ChunkSize;              // Each chunk covers ChunkSize columns
        int chunkStartY = chunk.y * ChunkSize;              // and ChunkSize rows

        int lx = threadIdx.x; 	                            //Local x index
        int ly = threadIdx.y;                               //Local y index
        int ll = ly * ChunkSize + lx;                       //Local linearized index
        
        int gy = chunkStartY + ly;                          //Global y index

        //Shared memory
        __shared__ int groupsChunk[ChunkSize * ChunkSize];  //Shared memory for groups of the local chunk
        __shared__ char cachedMat[ChunkSize * ChunkSize];   //Cached memory for the input matrix
        __shared__ bool blockStable;                        //Is the chunk in a stable configuration?
        __shared__ bool dirtyNeighbour;                     //Are we yet to account for changes in a neighbouring chunk?
        volatile __shared__ bool dirtyBlock;                //Has this chunk been changed?

        //Chess pattern
        int lxc = lx * 2 + (ly % 2);                        //Local chess x
        int gxc = chunkStartX + lxc;                        //Global chess x
        int lxc1 = lx * 2 + ((ly + 1) % 2);                 //Second local chess x
        int gxc1 = chunkStartX + lxc1;                      //Second global chess x

        //Initialize flags
        if (ll == 0) {
            blockStable = true;
            dirtyNeighbour = true;
            dirtyBlock = false;
        }

        //Vectorized access pattern
        int vlx = (lx << 1);                        //1-int x position in local space
        int vll = ly * ChunkSize + vlx;             //1-int linearized position in local space
        int vgx = vlx + chunkStartX;                //1-int x position in global space
        int vgl = gy * gpe + vgx;                   //1-int linearized position in global space
        int vglm = gy * mpe + vgx;                  //1-int linearized position in global space with input mat pitch

        //Init shared memory groups
        if (gy < height && vgx < width) {
            if (vgx + 1 < width) {
                reinterpret_cast<int2*>(groupsChunk)[vll >> 1] = reinterpret_cast<int2*>(groups)[vgl >> 1];
            } else {
                groupsChunk[vll] = groups[vgl];
            }
        }

        //Init shared matrix cache
        if (gy < height && vgx < width) {
            if (vgx + 1 < width) {
                reinterpret_cast<char2*>(cachedMat)[vll >> 1] = reinterpret_cast<const char2*>(mat)[vglm >> 1];
            } else {
                cachedMat[vll] = mat[vglm];
            }
        }


        __syncthreads(); //Await end of initialization

        /////////////////////// End of initialization ///////////////////////
        
        do { //Main propagation loop

            if (ll == 0) { //One thread resets variables
                blockStable = true;
                dirtyNeighbour = false;
            }
            checkDirty(ll, &dirtyNeighbour, status_matrix, numChunks, dirtyBlocks, spe, chunk);
            __syncthreads();

            propagate(lxc, ly, gxc, gy, width, height, cachedMat, groupsChunk, &blockStable, groups, dirtyNeighbour, mpe, gpe, mat);
            __syncthreads(); //Taking this off doesn't affect the correctness of the solution but causes unecessary propagations to happen worsening performance a bit
            propagate(lxc1, ly, gxc1, gy, width, height, cachedMat, groupsChunk, &blockStable, groups, dirtyNeighbour, mpe, gpe, mat);

            if (!blockStable) dirtyBlock = true;
            __syncthreads();
        } while (!blockStable);
        
        __threadfence();

        if (dirtyBlock) { //Write back to global memory and mark chunk as dirty
            
            if (gy < height && vgx < width) {
                if (vgx + 1 < width) {
                    reinterpret_cast<int2*>(groups)[vgl >> 1] = reinterpret_cast<int2*>(groupsChunk)[vll >> 1];
                } else {
                    groups[vgl] = groupsChunk[vll];
                }
            }

            if (ll == 0) {
                // Calculate valid neighbors. If a neighbour in one direction doesn't exist the dirty flag shoulnd't be raised because nothing
                // would be able to then lower it again
                int flags = WAITING;
                if (chunk.y > 0)             flags |= DIRTY_NORTH;
                if (chunk.x < numChunks.x-1) flags |= DIRTY_EAST;
                if (chunk.y < numChunks.y-1) flags |= DIRTY_SOUTH;
                if (chunk.x > 0)             flags |= DIRTY_WEST;
                //Atomically check if chunk was already dirty and make it dirty
                ChunkStatus old_status = (ChunkStatus)atomicOr( (int*)&status_matrix[chunk.y * spe + chunk.x], flags);
                if (old_status == 0) atomicAdd(dirtyBlocks, 1); // Only increment if previously clean
            }
        }

        __syncthreads(); //Wait for all threads before moving to next chunk

    }
    
}

GroupMatrix cuda_cc(const CharMatrix* __restrict__ mat) {

    // cudaEvent_t kernel_loop_start, kernel_loop_stop;
    // cudaEventCreate(&kernel_loop_start); cudaEventCreate(&kernel_loop_stop);
    // cudaEvent_t kernel_time_start, kernel_time_stop;
    // cudaEventCreate(&kernel_time_start); cudaEventCreate(&kernel_time_stop);

    dim3 numChunks( (mat->width + ChunkSize - 1) / ChunkSize, (mat->height + ChunkSize - 1) / ChunkSize );

    dim3 numBlocks( numChunks.x, numChunks.y );
    dim3 numThreads(ChunkSize/2, ChunkSize);

    //Initialize and allocate device memory for groups
    int* d_groups;
    size_t groups_pitch;
    GroupMatrix h_groups = initGroupsUnique(mat->width, mat->height);
    HANDLE_ERROR(cudaMallocPitch(&d_groups, &groups_pitch, mat->width * sizeof(int), mat->height));
    HANDLE_ERROR(cudaMemcpy2D(d_groups, groups_pitch, h_groups.groups, mat->width * sizeof(int), mat->width * sizeof(int), mat->height, cudaMemcpyHostToDevice));

    //Initialize and allocate device memory for character matrix
    char* d_mat;
    size_t mat_pitch;
    HANDLE_ERROR(cudaMallocPitch(&d_mat, &mat_pitch, mat->width * sizeof(char), mat->height));
    //Copy char matrix to device memory
    HANDLE_ERROR(cudaMemcpy2D(d_mat, mat_pitch, mat->matrix, mat->width * sizeof(char), mat->width * sizeof(char), mat->height, cudaMemcpyHostToDevice));

    //Initialize status matrix
    int statusSize = sizeof(ChunkStatus) * numChunks.x * numChunks.y;
    enum ChunkStatus* h_status_matrix;
    h_status_matrix = (ChunkStatus*)malloc(statusSize);
    // Initialize status matrix by forcing a first inter-block communication before anything else happens
    for (int x = 0; x < numChunks.x; x++) {
        for (int y = 0; y < numChunks.y; y++) {
            int status = WAITING;
            if (y < numChunks.y - 1) status |= DIRTY_SOUTH;
            if (x < numChunks.x - 1) status |= DIRTY_EAST;
            if (y > 0)               status |= DIRTY_NORTH;
            if (x > 0)               status |= DIRTY_WEST;
            h_status_matrix[y * numChunks.x + x] = (ChunkStatus)status;
        }
    }
    enum ChunkStatus* d_status_matrix;
    size_t status_pitch;
    HANDLE_ERROR(cudaMallocPitch(&d_status_matrix, &status_pitch, numChunks.x * sizeof(ChunkStatus), numChunks.y));
    HANDLE_ERROR(cudaMemcpy2D(d_status_matrix, status_pitch, h_status_matrix, numChunks.x * sizeof(ChunkStatus), numChunks.x * sizeof(ChunkStatus), numChunks.y, cudaMemcpyHostToDevice));

    //Dirty count
    int* d_dirty;
    int* h_dirty;
    cudaHostAlloc(&h_dirty, sizeof(int), 0);
    //cudaHostGetDevicePointer(&d_dirty, h_dirty, 0); Using mapped memory worsens performance considerably, better to manually handle data movement
    *h_dirty = numChunks.x * numChunks.y;
    HANDLE_ERROR(cudaMalloc((void**)&d_dirty, sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(d_dirty, h_dirty, sizeof(int), cudaMemcpyHostToDevice));

    int iters = 0;
    bool err = false;

    // float kernel_time = 0;

    //Loop until stable
    // cudaEventRecord(kernel_loop_start);
    while (*h_dirty > 0 && !err) {

        // printf("Dirty blocks: %d\n", *h_dirty);
        
        // cudaEventRecord(kernel_time_start);
        cc_kernel<<<numBlocks, numThreads>>>(d_groups, d_mat, mat->width, mat->height, d_status_matrix, numChunks, d_dirty, groups_pitch / sizeof(int), mat_pitch / sizeof(char), status_pitch / sizeof(ChunkStatus));
        // cudaLaunchCooperativeKernel((void*)cc_kernel, numBlocks, numThreads, args);
        // cudaEventRecord(kernel_time_stop);


        HANDLE_ERROR(cudaMemcpy(h_dirty, d_dirty, sizeof(int), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        // float t;
        // cudaEventElapsedTime(&t, kernel_time_start, kernel_time_stop);
        // kernel_time += t;
        
        checkCUDAError("call of cuda_cc kernel");

        iters++;
        if (iters > numChunks.x * numChunks.y * 100) { //Cap of 100 kernel iterations per chunk
            printf("Something went wrong! Quitting and logging partial solution\n");
            err = true;
        }

    }
    // cudaEventRecord(kernel_loop_stop);

    //Copy group matrix back to host
    HANDLE_ERROR(cudaMemcpy2D(h_groups.groups, mat->width * sizeof(int), d_groups, groups_pitch, mat->width * sizeof(int), mat->height, cudaMemcpyDeviceToHost));

    if (err) {
        //Dump groups and dirty matrix to a file
        GroupMatrix dirtyMatrix;
        dirtyMatrix.width = numChunks.x; dirtyMatrix.height = numChunks.y;
        dirtyMatrix.groups = (int*)malloc(numChunks.x * numChunks.y * sizeof(int));
        HANDLE_ERROR(cudaMemcpy2D(dirtyMatrix.groups, numChunks.x * sizeof(int), &d_status_matrix, status_pitch, numChunks.x * sizeof(int), numChunks.y, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        saveGroupMatrixToFile(&h_groups, "Outputs/Errors/err_groups.txt");
        saveGroupMatrixToFile(&dirtyMatrix, "Outputs/Errors/err_statuses.txt");
    }

    //Free device memory
    HANDLE_ERROR(cudaFree(d_groups));
    HANDLE_ERROR(cudaFree(d_mat));
    HANDLE_ERROR(cudaFree(d_dirty));
    HANDLE_ERROR(cudaFree(d_status_matrix));

    // float loop_time;
    // cudaEventElapsedTime(&loop_time, kernel_loop_start, kernel_loop_stop);
    // cudaEventDestroy(kernel_loop_start); cudaEventDestroy(kernel_loop_stop); cudaEventDestroy(kernel_time_start); cudaEventDestroy(kernel_time_stop);

    // printf("[Loop time: %.1fms | Kernel time: %.1fms]\t", loop_time, kernel_time);

    return h_groups;

}