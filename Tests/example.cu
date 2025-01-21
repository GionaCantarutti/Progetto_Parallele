#include <stdio.h>

__global__ void hello() {
    printf("Hello World!\n");
}

int main() {
    hello<<<1, 4>>>();

    printf("CPU says hi!\n");

    cudaDeviceSynchronize();

    return 0;
}