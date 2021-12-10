#include <cstdio>
#include "vector_arithmetic.cuh"
#include <vector>


cudaGraph_t buildQuadraticGraph(float * a, float *b, float *c, float *sol1, float *sol2, float *tmp, int numElements, int blockSize) {
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


int main(int argc, char *argv[]) {
    const int NUM_ELEMENTS = 64;
    const int BLOCK_SIZE = 32;
    
    float *a, *b, *c, *sol1, *sol2, *tmp;
    cudaMalloc(&a, sizeof(float)*NUM_ELEMENTS);
    cudaMalloc(&b, sizeof(float)*NUM_ELEMENTS);
    cudaMalloc(&c, sizeof(float)*NUM_ELEMENTS);
    cudaMalloc(&sol1, sizeof(float)*NUM_ELEMENTS);
    cudaMalloc(&sol2, sizeof(float)*NUM_ELEMENTS);
    cudaMalloc(&tmp, sizeof(float)*NUM_ELEMENTS);

    fillArray<<<NUM_ELEMENTS/BLOCK_SIZE + 1, BLOCK_SIZE>>>(a, NUM_ELEMENTS, 1);
    fillArray<<<NUM_ELEMENTS/BLOCK_SIZE + 1, BLOCK_SIZE>>>(b, NUM_ELEMENTS, 0);
    fillArray<<<NUM_ELEMENTS/BLOCK_SIZE + 1, BLOCK_SIZE>>>(c, NUM_ELEMENTS, -4);

    cudaGraph_t innerGraph = buildQuadraticGraph(a,b,c,sol1,sol2,tmp, NUM_ELEMENTS,BLOCK_SIZE);

    std::vector<cudaGraphNode_t> nodeDependencies;

    cudaEvent_t clockStart, clockStop;
    cudaEventCreate(&clockStart);
    cudaEventCreate(&clockStop);

    cudaGraphExec_t graphExecutable;

    int numTrials = 20;
    // Loop from 1to 1024, doubling each loop
    for (int numIterations=1; numIterations<=1024; numIterations<<=1) {
        cudaGraph_t builtinGraph;
        cudaGraphCreate(&builtinGraph, 0);

        nodeDependencies.clear();

        cudaGraphNode_t childNodes[numIterations];
        for (int i=0; i<numIterations; i++) {
            cudaGraphAddChildGraphNode(&(childNodes[i]), builtinGraph, nodeDependencies.data(), nodeDependencies.size(), innerGraph);
            nodeDependencies.clear();
            nodeDependencies.push_back(childNodes[i]);
        }

        cudaGraphInstantiate(&graphExecutable, builtinGraph, NULL, NULL, 0);

        printf("%d,",numIterations);

        for (int n=0; n<numTrials; n++) {

            cudaEventRecord(clockStart);

                for (int i=0; i<(131072/numIterations); i++)
                    cudaGraphLaunch(graphExecutable, 0);

            cudaEventRecord(clockStop);
            cudaEventSynchronize(clockStop);

            float timeElapsed;
            cudaEventElapsedTime(&timeElapsed, clockStart, clockStop);

            printf("%.4f",timeElapsed);
            if (numTrials-n==1) 
                printf("\n");
            else
                printf(",");
        }

        cudaGraphDestroy(builtinGraph);
        cudaGraphExecDestroy(graphExecutable);
    }

    cudaGraphDestroy(innerGraph);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(sol1);
    cudaFree(sol2);
    cudaFree(tmp);
}