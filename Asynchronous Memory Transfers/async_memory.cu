#include <cstdio>

__global__ void array_fill_1D(int *array, int arraySize, int value) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < arraySize)
        array[id] = value;
}

int main(int argc, char *argv[]) {
    const int NUM_ELEMENTS = 32;
    int block_size = 32;
    int grid_size = 1;

    int *h_array;
    cudaMallocHost(&h_array, sizeof(int)*NUM_ELEMENTS);

    int *d_array;
    cudaMalloc(&d_array, sizeof(int)*NUM_ELEMENTS);

    array_fill_1D<<<grid_size,block_size>>>(d_array, NUM_ELEMENTS, 1);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(h_array, d_array, sizeof(int)*NUM_ELEMENTS, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    for(int i=0; i<NUM_ELEMENTS; i++)
        printf("%d ",h_array[i]);

    cudaStreamDestroy(stream);
    cudaFree(d_array);
    cudaFreeHost(h_array);
}
