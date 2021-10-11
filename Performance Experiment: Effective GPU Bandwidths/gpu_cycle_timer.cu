#include <cstdio>

__global__ void localCopy(int *times, int* input, int* output) {
    int clockStart;

    int localInput[4];
    int localOutput[4];
    for (int i=0; i<4; i++) {
        localInput[i] = input[i];
    }

    clockStart = clock();
        for (int i=0; i<1024; i++) {
            localOutput[i%4] = i * localInput[i%4];
        }
    int clockStop = clock();

    for (int i=0; i<4; i++) {
        output[i] = localOutput[i];
    }

    if (threadIdx.x == 0)
        times[blockIdx.x] = clockStop - clockStart;
}

__global__ void sharedCopy(int *times, int) {
    int clockStart;

    clockStart = clock();
    __shared__ int shared[4];
    shared[0] = 1;
    for (int i=1; i<256; i++) {
        shared[i%4] = i * shared[(i-1)%4];
    }

    int clockStop = clock();

    if (threadIdx.x == 0)
    times[blockIdx.x] = clockStop - clockStart;
}

__global__ void localArrayCopy(int *times, int* input, int* output) {
    int clockStart;

    int localInput[4096];
    int localOutput[4096];
    for (int i=0; i<4096; i++) {
        localInput[i] = input[i];
    }

    clockStart = clock();
        for (int i=0; i<4096; i++) {
            localOutput[i] = i * localInput[i];
        }
    int clockStop = clock();

    for (int i=0; i<4096; i++) {
        output[i] = localOutput[i];
    }

    if (threadIdx.x == 0)
        times[blockIdx.x] = clockStop - clockStart;
}

int *timeLocalCopy(int num_blocks) {
    int *h_time;
    h_time = (int *) malloc(num_blocks*sizeof(int));

    

    int *d_time; 
    cudaMalloc(&d_time, num_blocks*sizeof(int));    

    
    // Time local copy and print times
    int h_local_in[4] = {1,2,3,4};

    int *d_local_in, *d_local_out;
    cudaMalloc(&d_local_in, 4*sizeof(int));
    cudaMalloc(&d_local_out, 4*sizeof(int));

    cudaMemcpy(d_local_in, h_local_in, 4*sizeof(int), cudaMemcpyHostToDevice);

    localCopy<<<num_blocks, 32>>>(d_time, d_local_in, d_local_out);

    cudaMemcpy(h_time, d_time, 80*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_time);
    cudaFree(d_local_in);
    cudaFree(d_local_out);
    return h_time;
}

int *timeLocalArrayCopy(int num_blocks) {
    int *h_time;
    h_time = (int *) malloc(num_blocks*sizeof(int));

    int *d_time; 
    cudaMalloc(&d_time, num_blocks*sizeof(int));    
    
    // Time local copy and print times
    int h_local_in[4096];
    for (int elem=0; elem<4096; elem++)
        h_local_in[elem] = elem;

    int *d_local_in, *d_local_out;
    cudaMalloc(&d_local_in, 4096*sizeof(int));
    cudaMalloc(&d_local_out, 4096*sizeof(int));

    cudaMemcpy(d_local_in, h_local_in, 4096*sizeof(int), cudaMemcpyHostToDevice);

    localCopy<<<num_blocks, 32>>>(d_time, d_local_in, d_local_out);

    cudaMemcpy(h_time, d_time, 80*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_time);
    cudaFree(d_local_in);
    cudaFree(d_local_out);
    return h_time;
}

int main(int argc, char *argv[]) {
    
    const int NUM_BLOCKS = 80;

    printf("memory_type,num_threads,num_blocks\n");

    int *local_times = timeLocalCopy(NUM_BLOCKS);
    printf("local,32,%d",NUM_BLOCKS);
    for (int i=0; i<NUM_BLOCKS; i++)
        printf(",%d",local_times[i]);
    printf("\n");

    free(local_times);

    int *local_array_times = timeLocalArrayCopy(NUM_BLOCKS);
    printf("local_array,32,%d",NUM_BLOCKS);
    for (int i=0; i<NUM_BLOCKS; i++)
        printf(",%d",local_array_times[i]);
    printf("\n");

    free(local_array_times);
}
