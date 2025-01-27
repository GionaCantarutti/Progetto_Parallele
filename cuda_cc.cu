#include "char_matrix.h"

#define ChunkSize 32

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


__global__ void cuda_cc(int** groups, char** mat, int width, int height) {

    int gx = blockIdx.x * blockDim.x + threadIdx.x; 	//Global x index
    int gy = blockIdx.y * blockDim.y + threadIdx.y;     //Global y index
    int gl = gy * width + gx;                           //Global linearized index
    int lx = threadIdx.x; 	                            //Local x index
    int ly = threadIdx.y;                               //Local y index
    int ll = gy * ChunkSize + gx;                       //Local linearized index

    __shared__ char groupsChunk[ChunkSize][ChunkSize];

    //Init groups
    groupsChunk[ly][lx] = ll;
    __syncthreads(); //Await end of initialization
    
    
}

GroupMatrix cuda_cc(CharMatrix* mat) {

    //Initialize and allocate device memory for groups
    int** d_groups;
    HANDLE_ERROR(cudaMalloc((void**)d_groups, mat->height * mat->width * sizeof(int)));

    //Initialize and allocate device memory for character matrix
    char** d_mat;
    HANDLE_ERROR(cudaMalloc((void**)d_mat, mat->height * mat->width * sizeof(char)));

    //Copy char matrix to device memory
    HANDLE_ERROR(cudaMemcpy(d_mat, (void*)mat->matrix, mat->width * mat->height * sizeof(char), cudaMemcpyHostToDevice));

    dim3 numBlocks(ceil(mat->width / ChunkSize), (mat->height) / ChunkSize);
    dim3 numThreads(ChunkSize, ChunkSize);
    cuda_cc<<<numBlocks, numThreads>>>(d_groups, d_mat, mat->width, mat->height);
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