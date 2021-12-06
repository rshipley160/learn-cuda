#include <cstdio>
#include "vector_arithmetic.cuh"

#define BLOCK_SIZE 512

float quadraticUsingEvents(int numElements, int iterations) {
    int gridSize = (numElements / BLOCK_SIZE) + 1;

    float *a, *b, *c, *sol1, *sol2, *tmp;
    cudaMalloc(&a, sizeof(float)*numElements);
    cudaMalloc(&b, sizeof(float)*numElements);
    cudaMalloc(&c, sizeof(float)*numElements);
    cudaMalloc(&sol1, sizeof(float)*numElements);
    cudaMalloc(&sol2, sizeof(float)*numElements);
    cudaMalloc(&tmp, sizeof(float)*numElements);

    fillArray<<<gridSize, BLOCK_SIZE>>>(a, numElements);
    fillArray<<<gridSize, BLOCK_SIZE>>>(b, numElements);
    fillArray<<<gridSize, BLOCK_SIZE>>>(c, numElements);

    cudaStream_t bMinus;
    cudaStream_t bPlus;
    cudaStreamCreate(&bMinus);
    cudaStreamCreate(&bPlus);

    cudaEvent_t bPlusComplete, bMinusComplete;
    cudaEventCreate(&bPlusComplete);
    cudaEventCreate(&bMinusComplete);

    cudaEvent_t clockStart, clockStop;
    cudaEventCreate(&clockStart);
    cudaEventCreate(&clockStop);

    // Warm up both streams before beginning timing
    elementwiseProduct<<<gridSize, BLOCK_SIZE, 0, bMinus>>>(b, b, sol1, numElements);
    elementScalarProduct<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(a, c, -4, sol2, numElements);
    cudaStreamSynchronize(bMinus);
    cudaStreamSynchronize(bPlus);

    cudaEventRecord(clockStart);

        for (int i=0; i<iterations; i++) { 

        // Concurrent
        elementwiseProduct<<<gridSize, BLOCK_SIZE, 0, bMinus>>>(b, b, sol1, numElements);
        elementScalarProduct<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(a, c, -4, sol2, numElements);

        // Use events to ensure completion
        cudaEventRecord(bMinusComplete, bMinus);
        cudaStreamWaitEvent(bPlus, bMinusComplete);

        elementwiseSum<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(sol1, sol2, sol1, numElements);
        elementwiseSqrt<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(sol1, tmp, numElements);

        // Sync again - must have determinant before proceeding
        cudaEventRecord(bPlusComplete, bPlus);
        cudaStreamWaitEvent(bMinus, bPlusComplete);

        elementwiseDifference<<<gridSize, BLOCK_SIZE, 0, bMinus>>>(b, tmp, sol1, numElements);
        elementwiseSum<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(b, tmp, sol2, numElements);

        elementwiseQuotient<<<gridSize, BLOCK_SIZE, 0, bMinus>>>(sol1, a, 0.5, sol1, numElements);
        elementwiseQuotient<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(sol2, a, 0.5, sol2, numElements);

        // Make sure that both streams are done before stopping timer
        cudaEventRecord(bMinusComplete, bMinus);
        cudaEventRecord(bPlusComplete, bPlus);
        cudaEventSynchronize(bMinusComplete);
        cudaEventSynchronize(bPlusComplete);

        }

    cudaEventRecord(clockStop);
    cudaEventSynchronize(clockStop);
    float timeElapsed;
    cudaEventElapsedTime(&timeElapsed, clockStart, clockStop);

    cudaEventDestroy(clockStart);
    cudaEventDestroy(clockStop);

    cudaStreamDestroy(bMinus);
    cudaStreamDestroy(bPlus);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(tmp);
    cudaFree(sol1);
    cudaFree(sol2);

    return timeElapsed;
}

float quadraticUsingStreamSync(int numElements, int iterations) {
    int gridSize = numElements / BLOCK_SIZE + 1;

    float *a, *b, *c, *sol1, *sol2, *tmp;
    cudaMalloc(&a, sizeof(float)*numElements);
    cudaMalloc(&b, sizeof(float)*numElements);
    cudaMalloc(&c, sizeof(float)*numElements);
    cudaMalloc(&sol1, sizeof(float)*numElements);
    cudaMalloc(&sol2, sizeof(float)*numElements);
    cudaMalloc(&tmp, sizeof(float)*numElements);

    fillArray<<<gridSize, BLOCK_SIZE>>>(a, numElements);
    fillArray<<<gridSize, BLOCK_SIZE>>>(b, numElements);
    fillArray<<<gridSize, BLOCK_SIZE>>>(c, numElements);

    cudaStream_t bMinus;
    cudaStream_t bPlus;
    cudaStreamCreate(&bMinus);
    cudaStreamCreate(&bPlus);

    cudaEvent_t clockStart, clockStop;
    cudaEventCreate(&clockStart);
    cudaEventCreate(&clockStop);

    // Warm up both streams before beginning timing
    elementwiseProduct<<<gridSize, BLOCK_SIZE, 0, bMinus>>>(b, b, sol1, numElements);
    elementScalarProduct<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(a, c, -4, sol2, numElements);
    cudaStreamSynchronize(bMinus);
    cudaStreamSynchronize(bPlus);

    cudaEventRecord(clockStart);

        for (int i=0; i<iterations; i++) { 

            // Concurrent
            elementwiseProduct<<<gridSize, BLOCK_SIZE, 0, bMinus>>>(b, b, sol1, numElements);
            elementScalarProduct<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(a, c, -4, sol2, numElements);

            // Sync streams to ensure these complete before next step
            cudaStreamSynchronize(bMinus);
            cudaStreamSynchronize(bPlus);

            elementwiseSum<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(sol1, sol2, sol1, numElements);

            elementwiseSqrt<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(sol1, tmp, numElements);

            // Sync again - must have determinant before proceeding
            cudaStreamSynchronize(bPlus);

            elementwiseDifference<<<gridSize, BLOCK_SIZE, 0, bMinus>>>(b, tmp, sol1, numElements);
            elementwiseSum<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(b, tmp, sol2, numElements);

            elementwiseQuotient<<<gridSize, BLOCK_SIZE, 0, bMinus>>>(sol1, a, 0.5, sol1, numElements);
            elementwiseQuotient<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(sol2, a, 0.5, sol2, numElements);

            // Make sure that both streams are done before stopping timer
            cudaStreamSynchronize(bPlus);
            cudaStreamSynchronize(bMinus);

        }

    cudaEventRecord(clockStop);
    cudaEventSynchronize(clockStop);
    float timeElapsed;
    cudaEventElapsedTime(&timeElapsed, clockStart, clockStop);

    cudaEventDestroy(clockStart);
    cudaEventDestroy(clockStop);

    cudaStreamDestroy(bMinus);
    cudaStreamDestroy(bPlus);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(tmp);
    cudaFree(sol1);
    cudaFree(sol2);

    return timeElapsed;
}

float quadraticUsingDeviceSync(int numElements, int iterations) {
    int gridSize = numElements / BLOCK_SIZE + 1;

    float *a, *b, *c, *sol1, *sol2, *tmp;
    cudaMalloc(&a, sizeof(float)*numElements);
    cudaMalloc(&b, sizeof(float)*numElements);
    cudaMalloc(&c, sizeof(float)*numElements);
    cudaMalloc(&sol1, sizeof(float)*numElements);
    cudaMalloc(&sol2, sizeof(float)*numElements);
    cudaMalloc(&tmp, sizeof(float)*numElements);

    fillArray<<<gridSize, BLOCK_SIZE>>>(a, numElements);
    fillArray<<<gridSize, BLOCK_SIZE>>>(b, numElements);
    fillArray<<<gridSize, BLOCK_SIZE>>>(c, numElements);

    cudaStream_t bMinus;
    cudaStream_t bPlus;
    cudaStreamCreate(&bMinus);
    cudaStreamCreate(&bPlus);

    cudaEvent_t clockStart, clockStop;
    cudaEventCreate(&clockStart);
    cudaEventCreate(&clockStop);

    // Warm up both streams before beginning timing
    elementwiseProduct<<<gridSize, BLOCK_SIZE, 0, bMinus>>>(b, b, sol1, numElements);
    elementScalarProduct<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(a, c, -4, sol2, numElements);
    cudaStreamSynchronize(bMinus);
    cudaStreamSynchronize(bPlus);

    cudaEventRecord(clockStart);

        for (int i=0; i<iterations; i++) { 
            // Concurrent
            elementwiseProduct<<<gridSize, BLOCK_SIZE, 0, bMinus>>>(b, b, sol1, numElements);
            elementScalarProduct<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(a, c, -4, sol2, numElements);

            // Sync device to ensure these complete before next step
            cudaDeviceSynchronize();

            elementwiseSum<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(sol1, sol2, sol1, numElements);

            elementwiseSqrt<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(sol1, tmp, numElements);

            // Sync again - must have determinant before proceeding
            cudaDeviceSynchronize();

            elementwiseDifference<<<gridSize, BLOCK_SIZE, 0, bMinus>>>(b, tmp, sol1, numElements);
            elementwiseSum<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(b, tmp, sol2, numElements);

            elementwiseQuotient<<<gridSize, BLOCK_SIZE, 0, bMinus>>>(sol1, a, 0.5, sol1, numElements);
            elementwiseQuotient<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(sol2, a, 0.5, sol2, numElements);

            // Make sure that both streams are done before stopping timer
            cudaDeviceSynchronize();
        }

    cudaEventRecord(clockStop);
    cudaEventSynchronize(clockStop);
    float timeElapsed;
    cudaEventElapsedTime(&timeElapsed, clockStart, clockStop);

    cudaEventDestroy(clockStart);
    cudaEventDestroy(clockStop);

    cudaStreamDestroy(bMinus);
    cudaStreamDestroy(bPlus);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(tmp);
    cudaFree(sol1);
    cudaFree(sol2);

    return timeElapsed;
}

int main(int argc, char *argv[]) {
    const int NUM_ELEMENTS = 16382;
    const int TRIALS = 20;

    printf("Device,Stream,Event\n");
    for (int i=0; i<TRIALS; i++) {
        printf("%.4f,",quadraticUsingDeviceSync(NUM_ELEMENTS, 1024));
        printf("%.4f,",quadraticUsingStreamSync(NUM_ELEMENTS, 1024));
        printf("%.4f\n",quadraticUsingEvents(NUM_ELEMENTS, 1024));
    }
}