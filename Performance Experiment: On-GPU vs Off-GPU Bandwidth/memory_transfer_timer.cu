%%cu
#include <cstdio>

// GPUs will vary on the max number of threads that can be in a block, but most can accomodate 512 per block
#define THREADS_PER_BLOCK 512

/*
    gpu-bandwidth bytes numTests
    bytes - number of megabytes (MiB) to transfer in each test
    numTests - the number of times to run each bandwidth test
*/

// Generates a unique ID for each thread to ensure array accesses are thread safe
__device__ int globalID()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

// Copies contents of one array to another using the GPU
__global__ void knlMemCopy_1D(int *inbound, int *outbound, long size)
{
    int uniqueID = globalID();

    if (uniqueID < size)
        outbound[uniqueID] = inbound[uniqueID];
}

float *kernelCopy(long transfer_size, int repetitions) {
    // Calculate the number of blocks to use - we want one thread per element (byte) copied
    int numBlocks =  transfer_size / THREADS_PER_BLOCK;

    // If there are more bytes than can be evenly divided by a block's worth of threads, add a block to cover the rest
    if (transfer_size % THREADS_PER_BLOCK > 0)
        numBlocks++;

    cudaEvent_t clockStart, clockStop;
    cudaEventCreate(&clockStart);
    cudaEventCreate(&clockStop);

    int *d_input, *d_output;
    cudaMalloc(&d_input, transfer_size*sizeof(int));
    cudaMalloc(&d_output, transfer_size*sizeof(int));
    

    float trialTime;
    float *results = (float *) malloc(sizeof(float)*repetitions);

    // Initial run through to avoid any cold-start outliers
    knlMemCopy_1D<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_output, transfer_size);

    for (int rep=0; rep < repetitions; rep++)
    {
        cudaEventRecord(clockStart, 0);

            knlMemCopy_1D<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_output, transfer_size);
        
        cudaEventRecord(clockStop, 0);
        cudaEventSynchronize(clockStop);
        cudaEventElapsedTime(&trialTime, clockStart, clockStop);

        results[rep] = trialTime;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return results;
}

float *memcpyDtoD(long transfer_size, int repetitions) {
	// Calculate the number of blocks to use - we want one thread per element (byte) copied
    int numBlocks =  transfer_size / THREADS_PER_BLOCK;

    // If there are more bytes than can be evenly divided by a block's worth of threads, add a block to cover the rest
    if (transfer_size % THREADS_PER_BLOCK > 0)
        numBlocks++;

    cudaEvent_t clockStart, clockStop;
    cudaEventCreate(&clockStart);
    cudaEventCreate(&clockStop);

    int *d_input, *d_output;
    cudaMalloc(&d_input, transfer_size*sizeof(int));
    cudaMalloc(&d_output, transfer_size*sizeof(int));
    

    float trialTime;
    float *results = (float *) malloc(sizeof(float)*repetitions);

    // Initial run through to avoid any cold-start outliers
    cudaMemcpy(d_output, d_input, transfer_size*sizeof(int), cudaMemcpyDeviceToDevice);

    for (int rep=0; rep < repetitions; rep++)
    {
        cudaEventRecord(clockStart, 0);

            cudaMemcpy(d_output, d_input, transfer_size*sizeof(int), cudaMemcpyDeviceToDevice);
        
        cudaEventRecord(clockStop, 0);
        cudaEventSynchronize(clockStop);
        cudaEventElapsedTime(&trialTime, clockStart, clockStop);

        results[rep] = trialTime;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return results;
}

float *pagedDtoH(long transfer_size, int repetitions) {
	// Calculate the number of blocks to use - we want one thread per element (byte) copied
    int numBlocks =  transfer_size / THREADS_PER_BLOCK;

    // If there are more bytes than can be evenly divided by a block's worth of threads, add a block to cover the rest
    if (transfer_size % THREADS_PER_BLOCK > 0)
        numBlocks++;

    cudaEvent_t clockStart, clockStop;
    cudaEventCreate(&clockStart);
    cudaEventCreate(&clockStop);

    int *d_input, *h_output;
    cudaMalloc(&d_input, transfer_size*sizeof(int));
    h_output = (int *) malloc(transfer_size*sizeof(int));
    

    float trialTime;
    float *results = (float *) malloc(sizeof(float)*repetitions);

    // Initial run through to avoid any cold-start outliers
    cudaMemcpy(h_output, d_input, transfer_size*sizeof(int), cudaMemcpyDeviceToDevice);

    for (int rep=0; rep < repetitions; rep++)
    {
        cudaEventRecord(clockStart, 0);

            cudaMemcpy(h_output, d_input, transfer_size*sizeof(int), cudaMemcpyDeviceToHost);
        
        cudaEventRecord(clockStop, 0);
        cudaEventSynchronize(clockStop);
        cudaEventElapsedTime(&trialTime, clockStart, clockStop);

        results[rep] = trialTime;
    }

    cudaFree(d_input);
    free(h_output);

    return results;
}

float *pinnedDtoH(long transfer_size, int repetitions) {
	// Calculate the number of blocks to use - we want one thread per element (byte) copied
    int numBlocks =  transfer_size / THREADS_PER_BLOCK;

    // If there are more bytes than can be evenly divided by a block's worth of threads, add a block to cover the rest
    if (transfer_size % THREADS_PER_BLOCK > 0)
        numBlocks++;

    cudaEvent_t clockStart, clockStop;
    cudaEventCreate(&clockStart);
    cudaEventCreate(&clockStop);

    int *d_input, *h_output;
    cudaMalloc(&d_input, transfer_size*sizeof(int));
    cudaMallocHost(&h_output, transfer_size*sizeof(int));
    

    float trialTime;
    float *results = (float *) malloc(sizeof(float)*repetitions);

    // Initial run through to avoid any cold-start outliers
    cudaMemcpy(h_output, d_input, transfer_size*sizeof(int), cudaMemcpyDeviceToDevice);

    for (int rep=0; rep < repetitions; rep++)
    {
        cudaEventRecord(clockStart, 0);

            cudaMemcpy(h_output, d_input, transfer_size*sizeof(int), cudaMemcpyDeviceToHost);
        
        cudaEventRecord(clockStop, 0);
        cudaEventSynchronize(clockStop);
        cudaEventElapsedTime(&trialTime, clockStart, clockStop);

        results[rep] = trialTime;
    }

    cudaFree(d_input);
    cudaFree(h_output);

    return results;
}

float *unifiedMemcpy(long transfer_size, int repetitions) {
	// Calculate the number of blocks to use - we want one thread per element (byte) copied
    int numBlocks =  transfer_size / THREADS_PER_BLOCK;

    // If there are more bytes than can be evenly divided by a block's worth of threads, add a block to cover the rest
    if (transfer_size % THREADS_PER_BLOCK > 0)
        numBlocks++;

    cudaEvent_t clockStart, clockStop;
    cudaEventCreate(&clockStart);
    cudaEventCreate(&clockStop);

    int *u_input, *u_output;
    cudaMallocManaged(&u_input, transfer_size*sizeof(int));
	cudaMallocManaged(&u_output, transfer_size*sizeof(int));
    

    float trialTime;
    float *results = (float *) malloc(sizeof(float)*repetitions);

    // Initial run through to avoid any cold-start outliers
	memcpy(u_output, u_input, transfer_size*sizeof(int));

    for (int rep=0; rep < repetitions; rep++)
    {
        cudaEventRecord(clockStart, 0);

		memcpy(u_output, u_input, transfer_size*sizeof(int));
        
        cudaEventRecord(clockStop, 0);
        cudaEventSynchronize(clockStop);
        cudaEventElapsedTime(&trialTime, clockStart, clockStop);

        results[rep] = trialTime;
    }

    cudaFree(u_input);
    cudaFree(u_output);

    return results;
}

int main(int argc, char *argv[])
{
    // if (argc < 3) {
    //     printf("Too few arguments supplied. Make sure to supply the amount of memory to transfer (in MiB) and the number of repetitions to perform.\n");
    //     return 1;
    // } 

    int num_mibibytes = 128;//atoi(argv[1]);
    int repetitions = 10;//atoi(argv[2]);

    const int MI_B = 1048576;    // One MiB or 2^20 bytes

    // Total number of bytes (integer elements) to copy
    long transfer_size = num_mibibytes * MI_B / sizeof(int);

	printf("type,size,unit,numAccesses");
    for (int rep=1; rep <= repetitions; rep++)
        printf(",run%d",rep);
    printf("\n");

    // Test #1: GPU kernel copy capablilty    
    float *kernelCopyTimes = kernelCopy(transfer_size, repetitions);

	printf("kernelCopy,%d,MB,2",num_mibibytes);
    for (int rep=0; rep < repetitions; rep++)
        printf(",%f",kernelCopyTimes[rep]);
    printf("\n");

    free(kernelCopyTimes);

	// Test #2: GPU-GPU memcpy capablilty    
    float *deviceMemcpyTimes = memcpyDtoD(transfer_size, repetitions);

	printf("memcpyDtoD,%d,MB,2",num_mibibytes);
    for (int rep=0; rep < repetitions; rep++)
        printf(",%f",deviceMemcpyTimes[rep]);
    printf("\n");

    free(deviceMemcpyTimes);

	// Test #3: Paged GPU-CPU memcpy capablilty    
    float *pagedMemcpyTimes = pagedDtoH(transfer_size, repetitions);

	printf("pagedDtoH,%d,MB,2",num_mibibytes);
    for (int rep=0; rep < repetitions; rep++)
        printf(",%f",pagedMemcpyTimes[rep]);
    printf("\n");

    free(pagedMemcpyTimes);
 
 	// Test #4: Pinned GPU-CPU memcpy capablilty    
    float *pinnedMemcpyTimes = pinnedDtoH(transfer_size, repetitions);

	printf("pinnedDtoH,%d,MB,2",num_mibibytes);
    for (int rep=0; rep < repetitions; rep++)
        printf(",%f",pinnedMemcpyTimes[rep]);
    printf("\n");

    free(pinnedMemcpyTimes);

	// Test #5: Unified memory copy capablilty    
    float *unifiedMemcpyTimes = unifiedMemcpy(transfer_size, repetitions);

	printf("unifiedMemcpy,%d,MB,2",num_mibibytes);
    for (int rep=0; rep < repetitions; rep++)
        printf(",%f",unifiedMemcpyTimes[rep]);
    printf("\n");

	free(unifiedMemcpyTimes);
}
