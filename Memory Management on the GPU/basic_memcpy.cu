#include <cstdio>

__global__ void addOne(int *array, int size) {
    if (threadIdx.x < size)
        array[threadIdx.x] += 1;
}

int main(int argc, char *argv[]) {
    // Allocate 32 integer array of paged memory
    int numElements = 32;
    int *h_array = (int *) malloc(sizeof(int)*numElements);

    // Alternative: page-locked memory
    // int *h_array;
    // cudaMallocHost(&h_array, sizeof(int)*numElements);

    // Allocate 32 integer array of device memory
    int *d_array;
    cudaMalloc(&d_array, sizeof(int)*numElements);

    // Initialize the array with elements 0, 1, ..., n-1
    for (int i = 0; i < numElements; i++)
        h_array[i] = i;

    printf("Initial array contents: ");
    for (int i = 0; i < numElements; i++)
        printf("%d ",h_array[i]);
    printf("\n");

    cudaMemcpy(d_array, h_array, sizeof(int)*numElements, cudaMemcpyHostToDevice);

    addOne<<<1,numElements>>>(d_array, numElements);

    cudaMemcpy(h_array, d_array, sizeof(int)*numElements, cudaMemcpyDeviceToHost);

    printf("Final array contents: ");
    for (int i = 0; i < numElements; i++)
        printf("%d ",h_array[i]);
    printf("\n");
}
