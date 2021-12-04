#include <cstdio>
#include "vector_arithmetic.cuh"

int main(int argc, char *argv[]) {
    const int NUM_ELEMENTS = 4096;
    const int BLOCK_SIZE = 512;
    int gridSize = NUM_ELEMENTS / BLOCK_SIZE + 1;

    float *a, *b, *c, *sol1, *sol2, *tmp;
    cudaMalloc(&a, sizeof(float)*NUM_ELEMENTS);
    cudaMalloc(&b, sizeof(float)*NUM_ELEMENTS);
    cudaMalloc(&c, sizeof(float)*NUM_ELEMENTS);
    cudaMalloc(&sol1, sizeof(float)*NUM_ELEMENTS);
    cudaMalloc(&sol2, sizeof(float)*NUM_ELEMENTS);
    cudaMalloc(&tmp, sizeof(float)*NUM_ELEMENTS);

    fillArray<<<gridSize, BLOCK_SIZE>>>(a, NUM_ELEMENTS);
    fillArray<<<gridSize, BLOCK_SIZE>>>(b, NUM_ELEMENTS);
    fillArray<<<gridSize, BLOCK_SIZE>>>(c, NUM_ELEMENTS);

    cudaStream_t bMinus;
    cudaStream_t bPlus;
    cudaStreamCreate(&bMinus);
    cudaStreamCreate(&bPlus);

    cudaEvent_t bPlusComplete;
    cudaEvent_t bMinusComplete;
    cudaEventCreate(&bPlusComplete);
    cudaEventCreate(&bMinusComplete);

    // Concurrent
    elementwiseProduct<<<gridSize, BLOCK_SIZE, 0, bMinus>>>(b, b, sol1, NUM_ELEMENTS);
    elementScalarProduct<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(a, c, -4, sol2, NUM_ELEMENTS);
    cudaEventRecord(bMinusComplete, bMinus); // Signal bMinus completion of product kernel

    // Wait on product calculations to complete
    cudaStreamWaitEvent(bPlus, bMinusComplete);
    elementwiseSum<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(sol1, sol2, sol1, NUM_ELEMENTS);

    elementwiseSqrt<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(sol1, tmp, NUM_ELEMENTS);
    cudaEventRecord(bPlusComplete, bPlus);
    
    // Concurrent, wait on sqrt before starting
    cudaStreamWaitEvent(bMinus, bPlusComplete);
    elementwiseDifference<<<gridSize, BLOCK_SIZE, 0, bMinus>>>(b, tmp, sol1, NUM_ELEMENTS);
    elementwiseSum<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(b, tmp, sol2, NUM_ELEMENTS);

    //Concurrent
    elementwiseQuotient<<<gridSize, BLOCK_SIZE, 0, bMinus>>>(sol1, a, 0.5, sol1, NUM_ELEMENTS);    // wait on last difference to complete before running
    elementwiseQuotient<<<gridSize, BLOCK_SIZE, 0, bPlus>>>(sol2, a, 0.5, sol2, NUM_ELEMENTS);    // wait on last sum to complete before running

    cudaEventDestroy(bPlusComplete);
    cudaEventDestroy(bMinusComplete);

    cudaStreamDestroy(bMinus);
    cudaStreamDestroy(bPlus);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(tmp);
    cudaFree(sol1);
    cudaFree(sol2);
}