#include "char_matrix.h"

#define ChunkSize 64    //Has to be divisible by 2

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
const int dx[] = {1, -1, 0, 0};
const int dy[] = {0, 0, 1, -1};
//Map delta index to cardinal direction (plus reversed direction)
//const int bd[] = {1 << 1, 1 << 3, 1 << 0, 1 << 2};
const int bdr[] = {1 << 3, 1 << 1, 1 << 2, 1 << 0};

__device__ void propagate(int lxc, int lyc, int gxc, int gyc, int width, int height, char** mat, int groupsChunk[ChunkSize][ChunkSize], bool* blockStable) {

    if (lxc >= ChunkSize) lxc -= ChunkSize; //Faster than modulo even though it diverges?
    if (gxc >= width || gyc >= height) return; //Bounds check

    for (int i = 0; i < 4; i++) { //Loop over 4 neighbours
        int nlxc = lxc + dx[i]; int ngxc = gxc + dx[i];
        int nlyc = lyc + dy[i]; int ngyc = gyc + dx[i];
        if (nlxc >= ChunkSize || nlyc >= ChunkSize || nlxc < 0 || nlyc < 0) continue;   //Bounds check (local)
        if (ngxc >= width || ngyc >= height || ngxc < 0 || ngyc < 0) continue;          //Bounds check (global)
        bool override = (mat[gyc][gxc] == mat[ngyc][ngxc]) && groupsChunk[lyc][lxc] < groupsChunk[nlyc][nlxc];
        if (override) {
            groupsChunk[lyc][lxc] = groupsChunk[nlyc][nlxc];
            blockStable = false;
        }
    }

}

//Propagate but can access global groups and propagate from that to the local groupsChunk
__device__ void g_to_l_propagate(int lxc, int lyc, int gxc, int gyc, int width, int height, char** mat, int groupsChunk[ChunkSize][ChunkSize], bool* blockStable, int** groups) {

    if (lxc >= ChunkSize) lxc -= ChunkSize; //Faster than modulo even though it diverges?
    if (gxc >= width || gyc >= height) return; //Bounds check

    for (int i = 0; i < 4; i++) { //Loop over 4 neighbours
        int ngxc = gxc + dx[i];
        int ngyc = gyc + dx[i];
        if (ngxc >= width || ngyc >= height || ngxc < 0 || ngyc < 0) continue;          //Bounds check (global)
        bool override = (mat[gyc][gxc] == mat[ngyc][ngxc]) && groupsChunk[lyc][lxc] < groups[ngyc][ngxc];
        if (override) {
            groupsChunk[lyc][lxc] = groups[ngyc][ngxc];
            blockStable = false;
        }
    }

}

__device__ void serialCheckDirty(int ll, bool* dirtyNeighbour, ChunkStatus** status_matrix, int* busyBlocks) {
    //Thread 0 checks for dirty neighbouring blocks
    if (ll == 0) {
        for (int i = 0; i < 4; i++) {
            *dirtyNeighbour = status_matrix[blockIdx.x + dx[i]][blockIdx.y + dy[i]] & bdr[i] > 0; //Check if neighbour is dirty in this direction
            if (*dirtyNeighbour) {
                status_matrix[blockIdx.x + dx[i]][blockIdx.y + dy[i]] ^ bdr[i]; //Turn off dirty bit in the corresponding direction
                if (status_matrix[blockIdx.x + dx[i]][blockIdx.y + dy[i]] == 0) atomicAdd(busyBlocks, -1);
                break;
            }
        }
    }
    __syncthreads();
}

//ToDo
__device__ bool areAllBlocksWaiting() {
    return false;
}

__global__ void cuda_cc(int** groups, char** mat, int width, int height, ChunkStatus** status_matrix, int* busyBlocks) {

    int gx = blockIdx.x * blockDim.x + threadIdx.x; 	//Global x index
    int gy = blockIdx.y * blockDim.y + threadIdx.y;     //Global y index
    int gl = gy * width + gx;                           //Global linearized index
    int lx = threadIdx.x; 	                            //Local x index
    int ly = threadIdx.y;                               //Local y index
    int ll = ly * ChunkSize + lx;                       //Local linearized index

    __shared__ int groupsChunk[ChunkSize][ChunkSize];
    __shared__ bool blockStable = true;
    __shared__ bool dirtyNeighbour = true;
    __shared__ bool dirtyBlock = false;

    //Bounds check
    if (gx >= width || gy >= height) return;

    //Init groups
    groupsChunk[ly][lx] = gl;
    groupsChunk[ly][lx+(ChunkSize/2)] = gl + (ChunkSize/2);

    __syncthreads(); //Await end of initialization

    while (true) {
    
        do {

            //ToDo: only check this every so often?
            if (!dirtyNeighbour) {
                serialCheckDirty(ll, &dirtyNeighbour, status_matrix, busyBlocks);
            }


            blockStable = true;

            //Chess pattern
            int lxc = lx * 2 + (ly % 2);    //Local chess x
            int gxc = gx + (lxc - lx);      //Global chess x

            if (!dirtyNeighbour) {
                propagate(lxc, ly, gxc, gy, width, height, mat, groupsChunk, &blockStable);
                propagate(lxc + 1, ly, gxc + 1, gy, width, height, mat, groupsChunk, &blockStable);
            } else {
                g_to_l_propagate(lxc, ly, gxc, gy, width, height, mat, groupsChunk, &blockStable, groups);
                g_to_l_propagate(lxc + 1, ly, gxc + 1, gy, width, height, mat, groupsChunk, &blockStable, groups);
                dirtyNeighbour = false;
            }

            if (!blockStable) dirtyBlock = true;

            __syncthreads(); //Sync all at the end of an iteration
        } while (!blockStable);
        
        if (dirtyBlock) {
            //Race conditions shoulnd't be a concern here
            groups[gy][gx] = groupsChunk[ly][lx];   //Copy stable chunk to global
            groups[gy][gx + (ChunkSize/2)] = groupsChunk[ly][lx + (ChunkSize/2)];
            __syncthreads();
            if (ll == 0) {
                status_matrix[blockIdx.y][blockIdx.x] = (ChunkStatus)(DIRTY_NORTH | DIRTY_EAST | DIRTY_SOUTH | DIRTY_WEST);
                atomicAdd(busyBlocks, 1);
            }
        }

        //ToDo: Busy wait for dirty neighbours or full stabilization event
        while (true) {
            serialCheckDirty(ll, &dirtyNeighbour, status_matrix, busyBlocks);
            if (dirtyNeighbour) break;
            
            //ToDo: this is wrong. Needs a better way to check if everything is done
            if (busyBlocks == 0) return;

        }

    }
    
}

GroupMatrix cuda_cc(CharMatrix* mat) {

    dim3 numBlocks( ceil(mat->width / ChunkSize), ceil(mat->height / ChunkSize) );
    dim3 numThreads(ChunkSize/2, ChunkSize);

    //Initialize and allocate device memory for groups
    int** d_groups;
    HANDLE_ERROR(cudaMalloc((void**)d_groups, mat->height * mat->width * sizeof(int)));

    //Initialize and allocate device memory for character matrix
    char** d_mat;
    HANDLE_ERROR(cudaMalloc((void**)d_mat, mat->height * mat->width * sizeof(char)));

    //Copy char matrix to device memory
    HANDLE_ERROR(cudaMemcpy(d_mat, (void*)mat->matrix, mat->width * mat->height * sizeof(char), cudaMemcpyHostToDevice));

    //Initialize status matrix
    int statusSize = sizeof(ChunkStatus) * numBlocks.x * numBlocks.y;
    enum ChunkStatus** h_status_matrix;
    h_status_matrix = (ChunkStatus**)malloc(statusSize);
    for (int x = 0; x < numBlocks.x; x++) {
        for (int y = 0; y < numBlocks.y; y++) {
            h_status_matrix[y][x] = (ChunkStatus)(DIRTY_SOUTH | DIRTY_EAST);
        }
    }
    enum ChunkStatus** d_stauts_matrix;
    HANDLE_ERROR(cudaMalloc((void**)&d_stauts_matrix, statusSize));
    HANDLE_ERROR(cudaMemcpy(&d_stauts_matrix, &h_status_matrix, statusSize, cudaMemcpyHostToDevice));

    //Busy count
    int d_busy;
    HANDLE_ERROR(cudaMalloc((void**)&d_busy, sizeof(int)));
    HANDLE_ERROR(cudaMemset(&d_busy, 0, sizeof(int)));

    cuda_cc<<<numBlocks, numThreads>>>(d_groups, d_mat, mat->width, mat->height, d_stauts_matrix, &d_busy);
    cudaDeviceSynchronize();

    checkCUDAError("call of cuda_cc kernel");
    
    //Copy group matrix back to host
    GroupMatrix h_groups = simpleInitGroups(mat->width, mat->height);
    HANDLE_ERROR(cudaMemcpy(h_groups.groups, (void*)d_groups, mat->width * mat->height * sizeof(int), cudaMemcpyDeviceToHost));

    //Free device memory
    HANDLE_ERROR(cudaFree(d_groups));
    HANDLE_ERROR(cudaFree(d_mat));

    return h_groups;

}