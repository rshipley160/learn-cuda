# include <cstdio>

// Sum the contents of an array to a single value
__global__ void block_add_reduce(int *array, int arraySize) {
    int id = threadIdx.x;

    for (int stride = blockDim.x; stride>0; stride >>= 1) {
        if (id < stride)
            array[id] += array[id + stride];

        __syncthreads();
    }
}

// Fills an array consecutively from 0 to N-1
// Assumes N threads are running the kernel in any configuration
__global__ void fill_array(int *output, int N) 
{
    int globalIndex = threadIdx.x + blockIdx.x*blockDim.x;

    if (globalIndex < N)
        output[globalIndex] = globalIndex;
}


int main(int argc, char *argv[])
{
    // Number of elements to generate and sum - must be <= half your GPU's per-block thread limit
    const int N = 512;
    int numThreads = N / 2 + (N % 2);

    int* d;
    cudaMalloc((void**)&d, N * sizeof(int));

    int result;

    // Initialize the array. We are using 2 blocks since numThreads is half of N
    fill_array<<<2, numThreads>>>(d, N);

    // Reduce the array using addition
    block_add_reduce<<<1, numThreads>>>(d, N);

    // We can check the sum since we know what the sum of consecutive elements will always be
    int sumCheck = (N-1) * N / 2;  
    
    cudaMemcpy(&result, d, sizeof(int), cudaMemcpyDeviceToHost);
    if (result == sumCheck)
    {
        printf("Reduction sum executed correctly.\n");
        return 0;
    }
    printf("Reduction sum incorrect.\nSum: %d,\nCorrect Sum: %d\n", result, sumCheck);
    
    return 2;
}