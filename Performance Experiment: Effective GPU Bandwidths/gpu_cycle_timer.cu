#include <cstdio>

__global__ void localCopy(int *times) {
    int clockStart;

    clockStart = clock();
    int local[4];
    local[0] = 1;
    for (int i=1; i<256; i++) {
        local[i%4] = i * local[(i-1)%4];
    }

    int clockStop = clock();

    if (threadIdx.x == 0)
    times[blockIdx.x] = clockStop - clockStart;
}

__global__ void localArrayCopy(int *times) {
    int clockStart;

    int local[256];
    local[0]=1;

    clockStart = clock();
    for (int i=1; i<256; i++) {
        
        local[i] = local[i-1] + i;
    }

    int clockStop = clock();

    if (threadIdx.x == 0)
    times[blockIdx.x] = clockStop - clockStart;
}

// __global__ void getLocalCopyTime(int *elapsedTime) {
//     *elapsedTime = localArrayCopy();
// }

int main(int argc, char *argv[]) {
    int *h_time, *d_time;
    h_time = (int *) malloc(80*sizeof(int));
    cudaMalloc(&d_time, 80*sizeof(int));
    
    localArrayCopy<<<80, 32>>>(d_time);

    cudaMemcpy(h_time, d_time, 80*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i=0; i<80; i++)
        printf("%d ",h_time[i]);
    //printf("20,480 1-byte local accesses took %d clock cycles\n", *h_time);

    cudaFree(d_time);
    free(h_time);
}
