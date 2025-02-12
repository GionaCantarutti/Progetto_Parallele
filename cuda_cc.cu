#include "char_matrix.h"
#include <cooperative_groups.h>

#define ChunkSize 32    //Has to be divisible by 2

using namespace cooperative_groups;

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
    NONE_DIRTY = 0,
    DIRTY_NORTH = 1 << 0,
    DIRTY_EAST = 1 << 1,
    DIRTY_SOUTH = 1 << 2,
    DIRTY_WEST = 1 << 3
};

//Neighbour deltas
__constant__ const int dx[] = {1, -1, 0, 0};
__constant__ const int dy[] = {0, 0, 1, -1};
//Map delta index to cardinal direction (and the reverse mapping)
__constant__ const int bd[] = {DIRTY_EAST, DIRTY_WEST, DIRTY_SOUTH, DIRTY_NORTH};
__constant__ const int bdr[] = {DIRTY_WEST, DIRTY_EAST, DIRTY_NORTH, DIRTY_SOUTH};

__forceinline__ __device__ int propagate_min(char centerChar, char neighChar, int centerGID, int neighGID) {

    int mask = -(int)(centerChar == neighChar); //Mask cases where chars are different
    int diff = centerGID - neighGID;
    int sign = diff >> 31;
    int minCandidate = centerGID - (diff & ~sign);

    return (minCandidate & mask) | (centerGID & ~mask);

}

__forceinline__ __device__ void propagate(  int lxc, int lyc, int gxc, int gyc, int width, int height, const char* __restrict__ mat, int groupsChunk[ChunkSize * ChunkSize],
                            bool* __restrict__ blockStable, const int* __restrict__ groups, ChunkStatus dirtyNeighbour, int mp, int gp) {

    if (gxc >= width || gyc >= height) return; //Bounds check

    #pragma unroll
    for (int i = 0; i < 4; i++) { //Loop over 4 neighbours
        int nlxc = lxc + dx[i]; int ngxc = gxc + dx[i];
        int nlyc = lyc + dy[i]; int ngyc = gyc + dy[i];

        bool useGlobal = dirtyNeighbour & bd[i];
        bool outOfLocalBounds = nlxc >= ChunkSize || nlyc >= ChunkSize || nlxc < 0 || nlyc < 0;

        //ToDo: maybe we can prevent the warp divergence caused here?
        if (!useGlobal && outOfLocalBounds) continue;   //Bounds check (local)
        if (ngxc >= width || ngyc >= height || ngxc < 0 || ngyc < 0) continue;          //Bounds check (global)

        int neighGID = useGlobal && outOfLocalBounds ? groups[ngyc * gp + ngxc] : groupsChunk[nlyc * ChunkSize + nlxc];

        int newGID = propagate_min(mat[gyc * mp + gxc], mat[ngyc * mp + ngxc], groupsChunk[lyc * ChunkSize + lxc], neighGID);
        if (newGID < groupsChunk[lyc * ChunkSize + lxc]) {
            groupsChunk[lyc * ChunkSize + lxc] = newGID;
            *blockStable = false;
        }

    }

}

//Check if there's a dirty neighbouring chunk. If so lower the corresponding directional dirty flag on that chunk
__forceinline__  __device__ void serialCheckDirty(int ll, ChunkStatus* __restrict__ dirtyNeighbour, ChunkStatus* __restrict__ status_matrix, dim3 numChunks, int* __restrict__ dirtyBlocks, int sp, dim3 chunk) {
    __threadfence(); //Prevent rare inconsistencies
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int nx = chunk.x + dx[i];    //New x value
        int ny = chunk.y + dy[i];    //New y value
        if (ny >= numChunks.y || ny < 0 || nx >= numChunks.x || nx < 0) continue; //Boundary check
        int index = ny * sp + nx;

        int mask = ~bdr[i];

        ChunkStatus old_status = (ChunkStatus)atomicAnd((int*)&status_matrix[index], mask);

        __threadfence(); //Prevent more rare inconsistencies

        if (old_status & bdr[i]) {
            *dirtyNeighbour = (ChunkStatus)(*dirtyNeighbour | bd[i]);
            if ((old_status & mask) == 0) { //If the last dirty bit has been cleared by this operation decrement dirtyBlocks
                atomicAdd(dirtyBlocks, -1);
            }
        }
    }
}

//ToDo: probs there's a way to get block count without passing it as argument
__global__ void cc_kernel(int* groups, const char* __restrict__ mat, int width, int height, ChunkStatus* __restrict__ status_matrix, dim3 numChunks, int* __restrict__ dirtyBlocks, size_t gpe, size_t mpe, size_t spe) {

    //Grid-stride loop
    for (int chunkIndex = blockIdx.y * gridDim.x + blockIdx.x; chunkIndex < numChunks.x * numChunks.y; chunkIndex += gridDim.y * gridDim.x) {
    
        //Each thread will handle two cells each (hence the doubled indexes). In the memory management part we split the 32x32 chunk into two 16x32 sections.
        //In the iterative algorithm part instead we split the 32x32 chunk into a chessboard pattern of alternating cells so that we can avoid race dontions.

        dim3 chunk(chunkIndex % numChunks.x, chunkIndex / numChunks.x);

        int blockStartX = chunk.x * ChunkSize;              // Each block covers ChunkSize columns
        int blockStartY = chunk.y * ChunkSize;              // and ChunkSize rows

        int lx = threadIdx.x; 	                            //Local x index
        int ly = threadIdx.y;                               //Local y index
        int ll = ly * ChunkSize + lx;                       //Local linearized index
        
        int gy = blockStartY + ly;                          //Global y index

        __shared__ int groupsChunk[ChunkSize * ChunkSize];  //Shared memory for groups of the local chunk
        __shared__ bool blockStable;                        //Is the chunk in a stable configuration?
        __shared__ ChunkStatus dirtyNeighbour;              //Are we yet to account for changes in a neighbouring chunk?
        volatile __shared__ bool dirtyBlock;                //Has this chunk been changed?

        //Chess pattern
        int lxc = lx * 2 + (ly % 2);    //Local chess x
        int gxc = blockStartX + lxc;    //Global chess x
        int lxc1 = lx * 2 + ((ly + 1) % 2);
        int gxc1 = blockStartX + lxc1;

        //Initialize flags
        if (ll == 0) {
            blockStable = true;
            dirtyNeighbour = NONE_DIRTY;
            dirtyBlock = false;
        }

        //Vectorized access pattern
        int vlx = (lx << 1);                        //1-int x position in local space
        int vll = ly * ChunkSize + vlx;             //1-int linearized position in local space
        int vgx = vlx + blockStartX;                //1-int x position in global space
        int vgl = gy * gpe + vgx;                    //1-int linearized position in global space

        //Clear garbage data by setting it to a safe value
        // groupsChunk[threadIdx.y * ChunkSize + threadIdx.x] = INT_MAX;
        // groupsChunk[threadIdx.y * ChunkSize + threadIdx.x + (ChunkSize/2)] = INT_MAX;
        // __syncthreads();

        //Init shared memory groups
        if (gy < height && vgx < width) {
            if (vgx + 1 < width) {
                reinterpret_cast<int2*>(groupsChunk)[vll >> 1] = reinterpret_cast<int2*>(groups)[vgl >> 1];
            } else {
                groupsChunk[vll] = groups[vgl];
            }
        }


        __syncthreads(); //Await end of initialization

        /////////////////////// End of initialization ///////////////////////
        
        do {

            if (ll == 0) {
                blockStable = true;
                dirtyNeighbour = NONE_DIRTY;
                //ToDo: only check this every so often?
                serialCheckDirty(ll, &dirtyNeighbour, status_matrix, numChunks, dirtyBlocks, spe, chunk);
            }
            __syncthreads();

            

            propagate(lxc, ly, gxc, gy, width, height, mat, groupsChunk, &blockStable, groups, dirtyNeighbour, mpe, gpe);
            __syncthreads(); //Taking this off doesn't affect the correctness of the solution but causes unecessary propagations to happen worsening performance a bit
            __threadfence_block();
            propagate(lxc1, ly, gxc1, gy, width, height, mat, groupsChunk, &blockStable, groups, dirtyNeighbour, mpe, gpe);

            //__syncthreads(); //Sync all at the end of an iteration
            __threadfence();
            if (!blockStable) dirtyBlock = true;
            __syncthreads();
        } while (!blockStable);
        
        __threadfence();

        if (dirtyBlock) {
            //Race conditions shoulnd't be a concern here
            if (gy < height && vgx < width) {
                if (vgx + 1 < width) {
                    reinterpret_cast<int2*>(groups)[vgl >> 1] = reinterpret_cast<int2*>(groupsChunk)[vll >> 1];
                } else {
                    groups[vgl] = groupsChunk[vll];
                }
            }

            __syncthreads(); //Only move on to marking as dirty once all writing is done
            __threadfence(); //Ensure that all threads see the write before the chunk can be marked as dirty

            if (ll == 0) {
                // Calculate valid neighbors. If a neighbour in one direction doesn't exist the dirty flag shoulnd't be raised because nothing
                // would be able to then lower it again
                int flags = NONE_DIRTY;
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
    dim3 numBlocks( numChunks.x, numChunks.y);
    dim3 numThreads(ChunkSize/2, ChunkSize);

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
    int statusSize = sizeof(ChunkStatus) * numChunks.x * numChunks.y;
    enum ChunkStatus* h_status_matrix;
    h_status_matrix = (ChunkStatus*)malloc(statusSize);
    // Initialize status matrix with by forcing a first inter-block communication before anything else happens
    for (int x = 0; x < numChunks.x; x++) {
        for (int y = 0; y < numChunks.y; y++) {
            int status = NONE_DIRTY;
            if (y < numChunks.y - 1) status |= DIRTY_SOUTH;
            if (x < numChunks.x - 1) status |= DIRTY_EAST;
            if (y > 0)               status |= DIRTY_NORTH;
            if (x > 0)               status |= DIRTY_WEST;
            h_status_matrix[y * numChunks.x + x] = (ChunkStatus)status;
        }
    }
    enum ChunkStatus* d_status_matrix;
    //HANDLE_ERROR(cudaMalloc((void**)&d_status_matrix, statusSize));
    //HANDLE_ERROR(cudaMemcpy(d_status_matrix, h_status_matrix, statusSize, cudaMemcpyHostToDevice));
    size_t status_pitch;
    HANDLE_ERROR(cudaMallocPitch(&d_status_matrix, &status_pitch, numChunks.x * sizeof(ChunkStatus), numChunks.y));
    HANDLE_ERROR(cudaMemcpy2D(d_status_matrix, status_pitch, h_status_matrix, numChunks.x * sizeof(ChunkStatus), numChunks.x * sizeof(ChunkStatus), numChunks.y, cudaMemcpyHostToDevice));

    //Dirty count
    int* d_dirty;
    int* h_dirty;
    cudaHostAlloc(&h_dirty, sizeof(int), 0);
    //cudaHostGetDevicePointer(&d_dirty, h_dirty, 0); Using mapped memory worsens performance considerably
    *h_dirty = numChunks.x * numChunks.y;
    HANDLE_ERROR(cudaMalloc((void**)&d_dirty, sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(d_dirty, h_dirty, sizeof(int), cudaMemcpyHostToDevice));

    int iters = 0;
    bool err = false;

    //float kernel_time = 0;

    //Loop until stable
    //cudaEventRecord(kernel_loop_start);
    while (*h_dirty > 0 && !err) {

        // printf("Dirty blocks: %d\n", *h_dirty);
        
        // int groups_pitch_e = groups_pitch / sizeof(int); int mat_pitch_e = mat_pitch / sizeof(char); int status_pitch_e = status_pitch / sizeof(ChunkStatus);
        // void* args[] = {&d_groups, &d_mat, (void*)&mat->width, (void*)&mat->height, &d_status_matrix, &numChunks, &d_dirty, &groups_pitch_e, &mat_pitch_e, &status_pitch_e};
        // void** args_dyn = (void**)malloc(sizeof(args));
        // memcpy(args_dyn, args, sizeof(args));
        
        //cudaEventRecord(kernel_time_start);
        cc_kernel<<<numBlocks, numThreads>>>(d_groups, d_mat, mat->width, mat->height, d_status_matrix, numChunks, d_dirty, groups_pitch / sizeof(int), mat_pitch / sizeof(char), status_pitch / sizeof(ChunkStatus));
        // cudaLaunchCooperativeKernel((void*)cc_kernel, numBlocks, numThreads, args_dyn);
        //cudaEventRecord(kernel_time_stop);


        HANDLE_ERROR(cudaMemcpy(h_dirty, d_dirty, sizeof(int), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        // float t;
        // cudaEventElapsedTime(&t, kernel_time_start, kernel_time_stop);
        // kernel_time += t;
        
        checkCUDAError("call of cuda_cc kernel");

        iters++;
        if (iters > numChunks.x * numChunks.y * 100) { //Cap of 100 kernel iterations per chunk
            printf("Something went wrong! Quitting and logging solution\n");
            err = true;
        }

    }
    //cudaEventRecord(kernel_loop_stop);

    //Copy group matrix back to host
    //HANDLE_ERROR(cudaMemcpy(h_groups.groups, (void*)d_groups, mat->width * mat->height * sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy2D(h_groups.groups, mat->width * sizeof(int), d_groups, groups_pitch, mat->width * sizeof(int), mat->height, cudaMemcpyDeviceToHost));

    if (err) {
        //Dump groups and dirty matrix to a file
        GroupMatrix dirtyMatrix;
        dirtyMatrix.width = numChunks.x; dirtyMatrix.height = numChunks.y;
        dirtyMatrix.groups = (int*)malloc(numChunks.x * numChunks.y * sizeof(int));
        //HANDLE_ERROR(cudaMemcpy(dirtyMatrix.groups, (void*)d_status_matrix, dirtyMatrix.width * dirtyMatrix.height * sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy2D(dirtyMatrix.groups, numChunks.x * sizeof(int), &d_status_matrix, status_pitch, numChunks.x * sizeof(int), numChunks.y, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize(); //ToDo: check if needed
        saveGroupMatrixToFile(&h_groups, "Outputs/Errors/err_groups.txt");
        saveGroupMatrixToFile(&dirtyMatrix, "Outputs/Errors/err_statuses.txt");
    }

    //Free device memory
    HANDLE_ERROR(cudaFree(d_groups));
    HANDLE_ERROR(cudaFree(d_mat));

    cudaDeviceSynchronize();

    // float loop_time;
    // cudaEventElapsedTime(&loop_time, kernel_loop_start, kernel_loop_stop);
    // cudaEventDestroy(kernel_loop_start); cudaEventDestroy(kernel_loop_stop); cudaEventDestroy(kernel_time_start); cudaEventDestroy(kernel_time_stop);

    // printf("[Loop time: %.1fms | Kernel time: %.1fms]\t", loop_time, kernel_time);

    return h_groups;

}