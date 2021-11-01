# include <cstdio>

// Returns a unique ID for a thread in a 1D kernel config
// Not unique for higher dimension kernels!
__device__ int unique_1D() {
    return blockDim.x * blockIdx.x + threadIdx.x;
}

// Show that the 1D thread ID function works by using each thread to label an array element
__global__ void id_threads_1D(int *array, int size) {
    int threadID = unique_1D();

    if (threadID < size)
        array[threadID] = threadID;
}

// Returns a unique ID for a thread in a 2D kernel config
// Not unique for higher dimension kernels!
__device__ int unique_2D() {
    int unique_ID = 0;

    int threadsPerGridRow = gridDim.x * blockDim.x * blockDim.y;
    unique_ID += threadsPerGridRow * blockIdx.y;

    int threadsPerBlock = blockDim.x * blockDim.y;
    unique_ID += threadsPerBlock * blockIdx.x;

    unique_ID += blockDim.x * threadIdx.y;

    unique_ID += threadIdx.x;

    return unique_ID;
}

// Show that the 2D thread ID function works by using each thread to label an array element
__global__ void id_threads_2D(int *array, int size) {
    int threadID = unique_2D();

    if (threadID < size)
        array[threadID] = threadID;
}

// Returns a unique ID for a thread in a 3D kernel config
__device__ int unique_3D() {
    int unique_ID = 0;

    int threadsPerGridSlice = blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y;
    unique_ID += threadsPerGridSlice * blockIdx.z;

    int threadsPerBlockSlice = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    unique_ID += threadsPerBlockSlice * threadIdx.z;

    unique_ID += unique_2D();

    return unique_ID;
}

// Show that the 3D thread ID function works by using each thread to label an array element
__global__ void id_threads_3D(int *array, int size) {
    int threadID = unique_3D();

    if (threadID < size)
        array[threadID] = threadID;
}

// Print an array, adding a newline after line_size elements
void array_print(int *array, int size, int line_size) {
    for (int i=0; i<size; i++) {
        printf("%2d ",array[i]);
        if (line_size - (i % line_size) == 1)
            printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int *h_array, *d_array;
    h_array = (int *) malloc(sizeof(int)*64);
    cudaMalloc(&d_array, sizeof(int)*64);

    // Launch 1D thread enumerator
    id_threads_1D<<<2, 32>>>(d_array, 64);

    cudaMemcpy(h_array, d_array, sizeof(int)*64, cudaMemcpyDeviceToHost);

    printf("Array after 1D enumerator:\n");
    array_print(h_array, 64, 16);

    cudaMemset(d_array, 0, sizeof(int)*64);

    printf("Array has been reset.\n\n");

    // Launch 2D thread enumerator
    dim3 threadsPerBlock2D = dim3(4, 4);
    dim3 blocksPerGrid2D   = dim3(2, 2);

    id_threads_2D<<<blocksPerGrid2D, threadsPerBlock2D>>>(d_array, 64);

    cudaMemcpy(h_array, d_array, sizeof(int)*64, cudaMemcpyDeviceToHost);

    printf("Array after 2D enumerator:\n");
    array_print(h_array, 64, 16);

    cudaMemset(d_array, 0, sizeof(int)*64);

    printf("Array has been reset.\n\n");

    // Launch 3D thread enumerator
    dim3 threadsPerBlock3D = dim3(2, 2, 2);
    dim3 blocksPerGrid3D   = dim3(2, 2, 2);

    id_threads_3D<<<blocksPerGrid2D, threadsPerBlock2D>>>(d_array, 64);

    cudaMemcpy(h_array, d_array, sizeof(int)*64, cudaMemcpyDeviceToHost);

    printf("Array after 3D enumerator:\n");
    array_print(h_array, 64, 16);

    free(h_array);
    cudaFree(d_array);
} 