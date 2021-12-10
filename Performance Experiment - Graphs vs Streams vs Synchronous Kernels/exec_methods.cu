#include <cstdio>
#include "vector_arithmetic.cuh"
#include <vector>

cudaGraph_t buildQuadraticExpGraph(float *a, float *b, float *c, float *sol1, float *sol2, float *tmp, int numElements, int blockSize) {
    int gridSize = numElements / blockSize + 1;

    cudaGraph_t quadraticGraph;
    cudaGraphCreate(&quadraticGraph, 0);

    // Build the nodes that are part of the graph
    cudaGraphNode_t bSquaredNode;
    cudaKernelNodeParams bSquaredParams = {0};
    bSquaredParams.blockDim = blockSize;
    bSquaredParams.gridDim = gridSize;
    bSquaredParams.func = (void *)elementwiseProduct;
    void *bSquaredfunc_params[4] = {(void *)&b, (void *)&b, (void *)&sol1, (void *) &numElements};
    bSquaredParams.kernelParams = (void **)bSquaredfunc_params;

    cudaGraphNode_t neg4acNode;
    cudaKernelNodeParams neg4acParams = {0};
    neg4acParams.blockDim = blockSize;
    neg4acParams.gridDim = gridSize;
    neg4acParams.func = (void *)elementScalarProduct;
    const float NEG_FOUR = -4.0;
    void *neg4acfunc_params[5] = {(void *)&a, (void *)&c, (void *)&NEG_FOUR, (void *)&sol2, (void *) &numElements};
    neg4acParams.kernelParams = (void **)neg4acfunc_params;

    cudaGraphNode_t determinantSumNode;
    cudaKernelNodeParams determinantSumParams = {0};
    determinantSumParams.blockDim = blockSize;
    determinantSumParams.gridDim = gridSize;
    determinantSumParams.func = (void *)elementwiseSum;
    void *determinantSumfunc_params[4] = {(void *)&sol1, (void *)&sol2, (void *)&sol1, (void *) &numElements};
    determinantSumParams.kernelParams = (void **)determinantSumfunc_params;

    cudaGraphNode_t determinantSqrtNode;
    cudaKernelNodeParams determinantSqrtParams = {0};
    determinantSqrtParams.blockDim = blockSize;
    determinantSqrtParams.gridDim = gridSize;
    determinantSqrtParams.func = (void *)elementwiseSqrt;
    void *determinantSqrtfunc_params[3] = {(void *)&sol1, (void *)&tmp, (void *) &numElements};
    determinantSqrtParams.kernelParams = (void **)determinantSqrtfunc_params;
    
    cudaGraphNode_t bPlusNode;
    cudaKernelNodeParams bPlusParams = {0};
    bPlusParams.blockDim = blockSize;
    bPlusParams.gridDim = gridSize;
    bPlusParams.func = (void *)elementwiseSum;
    void *bPlusfunc_params[4] = {(void *)&tmp, (void *)&b, (void *)&sol1, (void *) &numElements};
    bPlusParams.kernelParams = (void **)bPlusfunc_params;
    
    cudaGraphNode_t bMinusNode;
    cudaKernelNodeParams bMinusParams = {0};
    bMinusParams.blockDim = blockSize;
    bMinusParams.gridDim = gridSize;
    bMinusParams.func = (void *)elementwiseDifference;
    void *bMinusfunc_params[4] = {(void *)&b, (void *)&tmp,  (void *)&sol2, (void *) &numElements};
    bMinusParams.kernelParams = (void **)bMinusfunc_params;

    cudaGraphNode_t bPlusQuotientNode;
    cudaKernelNodeParams bPlusQuotientParams = {0};
    bPlusQuotientParams.blockDim = blockSize;
    bPlusQuotientParams.gridDim = gridSize;
    bPlusQuotientParams.func = (void *)elementwiseQuotient;
    const float ONE_HALF = 0.5;
    void *bPlusQuotientfunc_params[5] = {(void *)&sol1, (void *)&a, (void *)&ONE_HALF, (void *)&sol1, (void *) &numElements};
    bPlusQuotientParams.kernelParams = (void **)bPlusQuotientfunc_params;

    cudaGraphNode_t bMinusQuotientNode;
    cudaKernelNodeParams bMinusQuotientParams = {0};
    bMinusQuotientParams.blockDim = blockSize;
    bMinusQuotientParams.gridDim = gridSize;
    bMinusQuotientParams.func = (void *)elementwiseQuotient;
    void *bMinusQuotientfunc_params[5] = {(void *)&sol2, (void *)&a, (void *)&ONE_HALF, (void *)&sol2, (void *) &numElements};
    bMinusQuotientParams.kernelParams = (void **)bMinusQuotientfunc_params;

    std::vector<cudaGraphNode_t> nodeDependencies;

    cudaGraphAddKernelNode(&bSquaredNode, quadraticGraph, NULL, 0, &bSquaredParams);
    nodeDependencies.push_back(bSquaredNode);

    cudaGraphAddKernelNode(&neg4acNode, quadraticGraph, NULL, 0, &neg4acParams);
    nodeDependencies.push_back(neg4acNode);

    cudaGraphAddKernelNode(&determinantSumNode, quadraticGraph, nodeDependencies.data(), 2, &determinantSumParams);

    nodeDependencies.clear();
    nodeDependencies.push_back(determinantSumNode);

    cudaGraphAddKernelNode(&determinantSqrtNode, quadraticGraph, nodeDependencies.data(), 1, &determinantSqrtParams);

    nodeDependencies.clear();
    nodeDependencies.push_back(determinantSqrtNode);


    cudaGraphAddKernelNode(&bPlusNode, quadraticGraph, nodeDependencies.data(), 1, &bPlusParams);


    cudaGraphAddKernelNode(&bMinusNode, quadraticGraph, nodeDependencies.data(), 1, &bMinusParams);

    nodeDependencies.clear();
    nodeDependencies.push_back(bPlusNode);

    cudaGraphAddKernelNode(&bPlusQuotientNode, quadraticGraph, nodeDependencies.data(), 1, &bPlusQuotientParams);

    nodeDependencies.clear();
    nodeDependencies.push_back(bMinusNode);

    cudaGraphAddKernelNode(&bMinusQuotientNode, quadraticGraph, nodeDependencies.data(), 1, &bMinusQuotientParams);

    return quadraticGraph;
}

cudaGraph_t buildQuadraticCapGraph(float *a, float *b, float *c, float *sol1, float *sol2, float *tmp, int numElements, int blockSize) {
    int gridSize = numElements / blockSize + 1;

    cudaStream_t bMinus;
    cudaStream_t bPlus;
    cudaStreamCreate(&bMinus);
    cudaStreamCreate(&bPlus);

    cudaEvent_t bPlusComplete;
    cudaEvent_t bMinusComplete;
    cudaEventCreate(&bPlusComplete);
    cudaEventCreate(&bMinusComplete);

    cudaGraph_t quadraticGraph;
    cudaGraphCreate(&quadraticGraph, 0);

    cudaStreamBeginCapture(bMinus, cudaStreamCaptureModeGlobal);
        // Fork into bPlus to make stream capture record bPlus activity
        cudaEventRecord(bMinusComplete, bMinus);
        cudaStreamWaitEvent(bPlus, bMinusComplete);

        // Start graph activities
        elementwiseProduct<<<gridSize, blockSize, 0, bMinus>>>(b, b, sol1, numElements);
        elementScalarProduct<<<gridSize, blockSize, 0, bPlus>>>(a, c, -4, sol2, numElements);
        cudaEventRecord(bMinusComplete, bMinus);

        cudaStreamWaitEvent(bPlus, bMinusComplete);
        elementwiseSum<<<gridSize, blockSize, 0, bPlus>>>(sol1, sol2, sol1, numElements);

        elementwiseSqrt<<<gridSize, blockSize, 0, bPlus>>>(sol1, tmp, numElements);
        cudaEventRecord(bPlusComplete, bPlus);

        cudaStreamWaitEvent(bMinus, bPlusComplete);
        elementwiseDifference<<<gridSize, blockSize, 0, bMinus>>>(b, tmp, sol1, numElements);
        elementwiseSum<<<gridSize, blockSize, 0, bPlus>>>(b, tmp, sol2, numElements);

        elementwiseQuotient<<<gridSize, blockSize, 0, bMinus>>>(sol1, a, 0.5, sol1, numElements);
        elementwiseQuotient<<<gridSize, blockSize, 0, bPlus>>>(sol2, a, 0.5, sol2, numElements);

        // Join the bPlus stream back into bMinus
        cudaEventRecord(bPlusComplete, bPlus);
        cudaStreamWaitEvent(bMinus, bPlusComplete);

    cudaStreamEndCapture(bMinus, &quadraticGraph);

    cudaEventDestroy(bPlusComplete);
    cudaEventDestroy(bMinusComplete);

    cudaStreamDestroy(bMinus);
    cudaStreamDestroy(bPlus);

    return quadraticGraph;
}

float timeGraph(cudaGraph_t graph, int iterations) {
    cudaEvent_t clockStart, clockStop;
    cudaEventCreate(&clockStart);
    cudaEventCreate(&clockStop);

    cudaGraphExec_t executable;
    
    cudaGraphInstantiate(&executable, graph, NULL, NULL, 0);

    cudaEventRecord(clockStart);
        for(int i=0; i<iterations; i++)
        cudaGraphLaunch(executable, 0);
    cudaEventRecord(clockStop);
    cudaEventSynchronize(clockStop);

    float timeElapsed;
    cudaEventElapsedTime(&timeElapsed, clockStart, clockStop);

    cudaEventDestroy(clockStart);
    cudaEventDestroy(clockStop);

    return timeElapsed;
}

float quadraticUsingEvents(float *a, float *b, float *c, float *sol1, float *sol2, float* tmp, int numElements, int blockSize, int iterations) {
    int gridSize = (numElements / blockSize) + 1;

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
    elementwiseProduct<<<gridSize, blockSize, 0, bMinus>>>(b, b, sol1, numElements);
    elementScalarProduct<<<gridSize, blockSize, 0, bPlus>>>(a, c, -4, sol2, numElements);
    cudaStreamSynchronize(bMinus);
    cudaStreamSynchronize(bPlus);

    cudaEventRecord(clockStart);

        for (int i=0; i<iterations; i++) { 

            // Concurrent
            elementwiseProduct<<<gridSize, blockSize, 0, bMinus>>>(b, b, sol1, numElements);
            elementScalarProduct<<<gridSize, blockSize, 0, bPlus>>>(a, c, -4, sol2, numElements);

            // Use events to ensure completion
            cudaEventRecord(bMinusComplete, bMinus);
            cudaStreamWaitEvent(bPlus, bMinusComplete);

            elementwiseSum<<<gridSize, blockSize, 0, bPlus>>>(sol1, sol2, sol1, numElements);
            elementwiseSqrt<<<gridSize, blockSize, 0, bPlus>>>(sol1, tmp, numElements);

            // Sync again - must have determinant before proceeding
            cudaEventRecord(bPlusComplete, bPlus);
            cudaStreamWaitEvent(bMinus, bPlusComplete);

            elementwiseDifference<<<gridSize, blockSize, 0, bMinus>>>(b, tmp, sol1, numElements);
            elementwiseSum<<<gridSize, blockSize, 0, bPlus>>>(b, tmp, sol2, numElements);

            elementwiseQuotient<<<gridSize, blockSize, 0, bMinus>>>(sol1, a, 0.5, sol1, numElements);
            elementwiseQuotient<<<gridSize, blockSize, 0, bPlus>>>(sol2, a, 0.5, sol2, numElements);
        }

    // Make sure that both streams are done before stopping timer
    cudaEventRecord(bMinusComplete, bMinus);
    cudaEventRecord(bPlusComplete, bPlus);
    cudaEventSynchronize(bMinusComplete);
    cudaEventSynchronize(bPlusComplete);

    cudaEventRecord(clockStop);
    cudaEventSynchronize(clockStop);
    float timeElapsed;
    cudaEventElapsedTime(&timeElapsed, clockStart, clockStop);

    cudaEventDestroy(clockStart);
    cudaEventDestroy(clockStop);

    cudaStreamDestroy(bMinus);
    cudaStreamDestroy(bPlus);

    return timeElapsed;
}

float synchronousQuadratic(float *a, float *b, float *c, float *sol1, float *sol2, float *tmp, int numElements, int blockSize, int iterations) {
    int gridSize = (numElements / blockSize) + 1;

    cudaEvent_t clockStart, clockStop;
    cudaEventCreate(&clockStart);
    cudaEventCreate(&clockStop);

    // Warm up before beginning timing
    elementwiseProduct<<<gridSize, blockSize>>>(b, b, sol1, numElements);

    cudaEventRecord(clockStart);

        for (int i=0; i<iterations; i++) { 

            elementwiseProduct<<<gridSize, blockSize>>>(b, b, sol1, numElements);
            elementScalarProduct<<<gridSize, blockSize>>>(a, c, -4, sol2, numElements);

            elementwiseSum<<<gridSize, blockSize>>>(sol1, sol2, sol1, numElements);
            elementwiseSqrt<<<gridSize, blockSize>>>(sol1, tmp, numElements);

            elementwiseDifference<<<gridSize, blockSize>>>(b, tmp, sol1, numElements);
            elementwiseSum<<<gridSize, blockSize>>>(b, tmp, sol2, numElements);

            elementwiseQuotient<<<gridSize, blockSize>>>(sol1, a, 0.5, sol1, numElements);
            elementwiseQuotient<<<gridSize, blockSize>>>(sol2, a, 0.5, sol2, numElements);
        }

    cudaEventRecord(clockStop);
    cudaEventSynchronize(clockStop);
    float timeElapsed;
    cudaEventElapsedTime(&timeElapsed, clockStart, clockStop);

    cudaEventDestroy(clockStart);
    cudaEventDestroy(clockStop);

    return timeElapsed;
}

int main(int argc, char *argv[]) {
    const int NUM_ELEMENTS = 64;
    const int BLOCK_SIZE = 32;
    const int NUM_TRIALS = 20;
    const int ITERATIONS = 131072;

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

    cudaGraph_t expGraph = buildQuadraticExpGraph(a, b, c,sol1, sol2, tmp, NUM_ELEMENTS, BLOCK_SIZE);
    cudaGraph_t capGraph = buildQuadraticCapGraph(a, b, c,sol1, sol2, tmp, NUM_ELEMENTS, BLOCK_SIZE); 

    printf("Synchronous,Stream,Captured Graph,Explicit Graph\n");
    for (int i=0; i<NUM_TRIALS; i++) {
        float syncTime = synchronousQuadratic(a, b, c, sol1, sol2, tmp, NUM_ELEMENTS, BLOCK_SIZE, ITERATIONS);
        printf("%.4f,", syncTime);
        float streamTime = quadraticUsingEvents(a, b, c, sol1, sol2, tmp, NUM_ELEMENTS, BLOCK_SIZE, ITERATIONS);
        printf("%.4f,", streamTime);
        float capGraphTime = timeGraph(capGraph, ITERATIONS);
        printf("%.4f,", capGraphTime);
        float expGraphTime = timeGraph(expGraph, ITERATIONS);
        printf("%.4f\n", expGraphTime);
    }

    cudaGraphDestroy(expGraph);
    cudaGraphDestroy(capGraph);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(tmp);
    cudaFree(sol1);
    cudaFree(sol2);
}