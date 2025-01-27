#include "char_matrix.h"

#define NumThPerBlock 256
#define NumBlocks 1

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

}

GroupMatrix cuda_cc(CharMatrix* mat) {

    //Initialize and allocate device memory for groups
    int** h_group_ptrs = (int**)malloc(mat->width * sizeof(int*));
    int** d_groups;
    for (int i = 0; i < mat->width; i++) {
        HANDLE_ERROR(cudaMalloc((void**)&h_group_ptrs[i], mat->height * sizeof(int)));
    }
    HANDLE_ERROR(cudaMalloc((void**)&d_groups, mat->width * sizeof(int*)));
    HANDLE_ERROR(cudaMemcpy(d_groups, h_group_ptrs, mat->width * sizeof(int*), cudaMemcpyHostToDevice));
    free(h_group_ptrs);

    //Initialize and allocate device memory for character matrix
    char** h_mat_ptrs = (char**)malloc(mat->width * sizeof(char*));
    char** d_mat;
    for (int i = 0; i < mat->width; i++) {
        HANDLE_ERROR(cudaMalloc((void**)&h_mat_ptrs[i], mat->height * sizeof(char)));
    }
    HANDLE_ERROR(cudaMalloc((void**)&d_mat, mat->width * sizeof(char*)));
    HANDLE_ERROR(cudaMemcpy(d_mat, h_mat_ptrs, mat->width * sizeof(char*), cudaMemcpyHostToDevice));
    free(h_mat_ptrs);

    //Copy char matrix to device memory
    for (int i = 0; i < mat->width; i++) {
        HANDLE_ERROR(cudaMemcpy(d_mat[i], (void*)mat->matrix[i], mat->height * sizeof(char), cudaMemcpyHostToDevice));
    }

    cuda_cc<<<NumBlocks, NumThPerBlock>>>(d_groups, d_mat, mat->width, mat->height);

    cudaDeviceSynchronize();

    checkCUDAError("call of cuda_cc kernel");
    
    //Copy group matrix back to host
    GroupMatrix h_groups = simpleInitGroups(mat->width, mat->height);
    for (int i = 0; i < mat->width; i++) {
        HANDLE_ERROR(cudaMemcpy(h_groups.groups[i], (void*)d_groups[i], mat->height * sizeof(int), cudaMemcpyDeviceToHost));
    }

    //Free device memory
    for (int i = 0; i < mat->width; i++) {
        HANDLE_ERROR(cudaFree(d_groups[i]));
        HANDLE_ERROR(cudaFree(d_mat[i]));
    }
    HANDLE_ERROR(cudaFree(d_groups));
    HANDLE_ERROR(cudaFree(d_mat));

    return h_groups;

}