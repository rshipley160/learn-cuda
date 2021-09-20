#include <cstdio>

__global__ void fillWithOnes(int *array, int size) {
    if (threadIdx.x < size)
        array[threadIdx.x] = 1;
}

int main(int argc, char *argv[]) {
    int arraySize = 256;

    int *d_array;
    cudaMalloc(&d_array, sizeof(int)*arraySize);

    int *h_array;
    h_array = (int *) malloc(sizeof(int)*arraySize);

    printf("Contents before running kernel:\n");
    for (int element = 0; element < arraySize; element++) {
        printf("%d ", h_array[element]);

        // Add a line break after every 16 elements to create a 16 x 16 grid
        if (element % 16 == 15)
            printf("\n");
    }

    fillWithOnes<<<1, 256>>>(d_array, arraySize);

    cudaMemcpy(h_array, d_array, sizeof(int)*arraySize, cudaMemcpyDeviceToHost);

    printf("\nContents after running kernel:\n");
    for (int element = 0; element < arraySize; element++) {
        printf("%d ", h_array[element]);

        // Add a line break after every 16 elements to create a 16 x 16 grid
        if (element % 16 == 15)
            printf("\n");
    }
}
