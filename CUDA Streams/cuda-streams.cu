#include <cstdio>

// Fill an array with a given value
__global__ void array_fill_1D(int *array, int arraySize, int value) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    // Protect against accesses outside of array bounds
    if (id < arraySize)
        array[id] = value;
}

float sync_twinArrayFill(int *h_array, int * d_array, int arraySize, int value) {
    cudaEvent_t clockStart, clockStop;
    cudaEventCreate(&clockStart);
    cudaEventCreate(&clockStop);

    const int BLOCK_SIZE = 512;

    // Determine how many 512-thread blocks are needed to fill the array
    int gridSize = arraySize / BLOCK_SIZE;
    // Add an extra block if arraySize isn't evenly divisible by BLOCK_SIZE
    gridSize += (arraySize % BLOCK_SIZE) ? 1 : 0;

    cudaEventRecord(clockStart, 0);

        array_fill_1D<<<gridSize, BLOCK_SIZE>>>(d_array, arraySize, value);

        for (int i=0; i<arraySize; i++)
            h_array[i] = value;

    cudaEventRecord(clockStop, 0);

    cudaEventSynchronize(clockStop);
    
    float timeElapsed;
    cudaEventElapsedTime(&timeElapsed, clockStart, clockStop);

    cudaEventDestroy(clockStart);
    cudaEventDestroy(clockStop);

    return timeElapsed;
}

float async_twinArrayFill(int *h_array, int * d_array, int arraySize, int value) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t clockStart, clockStop;
    cudaEventCreate(&clockStart);
    cudaEventCreate(&clockStop);

    const int BLOCK_SIZE = 512;

    // Determine how many 512-thread blocks are needed to fill the array
    int gridSize = arraySize / BLOCK_SIZE;
    // Add an extra block if arraySize isn't evenly divisible by BLOCK_SIZE
    gridSize += (arraySize % BLOCK_SIZE) ? 1 : 0;

    cudaEventRecord(clockStart, 0);

        array_fill_1D<<<gridSize, BLOCK_SIZE, 0, stream>>>(d_array, arraySize, value);

        for (int i=0; i<arraySize; i++)
            h_array[i] = value;

        cudaStreamSynchronize(stream);

    cudaEventRecord(clockStop, 0);

    cudaEventSynchronize(clockStop);
    
    float timeElapsed;
    cudaEventElapsedTime(&timeElapsed, clockStart, clockStop);

    cudaEventDestroy(clockStart);
    cudaEventDestroy(clockStop);

    cudaStreamDestroy(stream);

    return timeElapsed;
}

int main(int argc, char *argv[]) {

    const int NUM_ELEMENTS = 1048576;

    int *h_array = (int *) malloc(sizeof(int)*NUM_ELEMENTS);
    
    int *d_array;
    cudaMalloc(&d_array, sizeof(int) * NUM_ELEMENTS);

    float sync_time = sync_twinArrayFill(h_array, d_array, NUM_ELEMENTS, 16);
    printf("Synchronous completion time: %f\n", sync_time);
    
    float async_time = async_twinArrayFill(h_array, d_array, NUM_ELEMENTS, 16);
    printf("Asynchronous completion time: %f\n", async_time);

    free(h_array);
    cudaFree(d_array);
}