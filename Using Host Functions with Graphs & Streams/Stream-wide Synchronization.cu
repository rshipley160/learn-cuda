#include <stdio.h>

// Step 1
/*
    Fills the remainders array with the results of n % divisor for n from 0 to maxDividend - 1
    Assumes a 1-dimensional block and grid configuration
*/
__global__ void knModulo(int divisor, int *remainders, int maxDividend) {
    // The unique rank of each thread
    int rank = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread guard to prevent illegal memory accesses
    if (rank > maxDividend)
        return;

    // If a rank is equal to the prime divisor, we don't want a value of 0 to be stored
    // because that would indicate that the rank is not prime when it actually is
    if (rank == divisor) {
        // Thus we make sure to store a nonzero value
        remainders[rank] = 1;
    }
    // We do not want to count 1 as a prime - if we do, the second pass won't register any prime numbers
    else if (rank == 1)
        remainders[rank] = 0;

    else
        remainders[rank] = rank % divisor;
}


// Step 2
// Calculates element-wise product of two arrays and stores results in dstArray
__global__ void knDotProduct(int *dstArray, int *otherArray, int size) {
    // The unique rank of each thread
    int rank = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread guard to prevent illegal memory accesses
    if (rank > size)
        return;

    dstArray[rank] = dstArray[rank] * otherArray[rank];
}


// Step 3
/*
    Stores all of the indices of non-zero elements in srcArray in destArray
    Returns the number of non-zero elements found
    destArray must be the same size as srcArray - could result in index out of bounds otherwise
*/
int getNonZeroElements(int *srcArray, int srcSize, int *destArray) {
    int destSize = 0;
    for(int srcIndex = 0; srcIndex < srcSize; srcIndex++)
    {
        if (srcArray[srcIndex] != 0)
        {
            destArray[destSize] = srcIndex;
            destSize = destSize + 1;
        }
    }
    return destSize;
}


int main(int argc, char *argv[]) {
    // The number of threads to launch our kernel with
    // Also the upper bound of our primes search
    const int NUM_THREADS = 512;

    // Our initial list of prime numbers < 10
    int numPrimes = 4;
    int h_primes[numPrimes] = {2, 3, 5, 7};

    // Allocate device memory for modulo operation remainders for each of our initial prime numbers
    int **d_modulo_remainders;
    d_modulo_remainders = new int *[4];
    for (int i = 0; i < 4; i++)
        cudaMalloc(&(d_modulo_remainders[i]), sizeof(int)*NUM_THREADS);

    // Allocate host remainder array to hold results of dot product in step 2
    int *h_modulo_remainders = (int *) malloc(sizeof(int)*NUM_THREADS);

    // Create & initialize one stream for each of the initial primes
    cudaStream_t streams[4];
    for (int i = 0; i < 4; i++)
        cudaStreamCreate(&streams[i]);

    for (int primeIndex = 0; primeIndex < numPrimes; primeIndex++)
        knModulo<<<1, NUM_THREADS, 0, streams[primeIndex]>>>(h_primes[primeIndex], d_modulo_remainders[primeIndex], NUM_THREADS);

    for (int step=1; step < numPrimes; step <<=1) {
        for (int streamIdx=0; streamIdx < numPrimes; streamIdx += step * 2) {
            int destStreamIdx = streamIdx;
            int srcStreamIdx  = streamIdx + step;
            if (srcStreamIdx < numPrimes) {
                // Make sure both streams have completed
                cudaStreamSynchronize(streams[destStreamIdx]);
                cudaStreamSynchronize(streams[srcStreamIdx]);
                
                // Run the dot product kernel on the destination stream
                knDotProduct<<<1, NUM_THREADS, 0, streams[destStreamIdx]>>>(d_modulo_remainders[destStreamIdx], d_modulo_remainders[srcStreamIdx], NUM_THREADS);
            }
        }
    }

    cudaStreamSynchronize(streams[0]);

    cudaMemcpy(h_modulo_remainders, d_modulo_remainders[0], sizeof(int)*NUM_THREADS, cudaMemcpyDeviceToHost);

    int *firstPassResults = (int *) malloc(sizeof(int)*NUM_THREADS);
    int numPrimes_new = getNonZeroElements(h_modulo_remainders, NUM_THREADS, firstPassResults);

    // Create a clean array to hold our new list of primes
    int primes_new[numPrimes_new];
    memcpy(primes_new, firstPassResults, sizeof(int)*numPrimes_new);

    // Now we no longer need any first pass data structures
    free(firstPassResults);
    free(h_modulo_remainders);
    for (int i=0; i<4; i++) {
        cudaFree(d_modulo_remainders[i]);
        cudaStreamDestroy(streams[i]);
    }
}
