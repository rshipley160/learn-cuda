#include <cstdio>
#include "vector_arithmetic.cuh"
#include <vector>

int main(int argc, char *argv[]) {
    const int NUM_ELEMENTS = 64;
    const int BLOCK_SIZE = 32;
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

    cudaGraph_t quadraticGraph;
    cudaGraphCreate(&quadraticGraph, 0);

    // Build the nodes that are part of the graph
    cudaGraphNode_t bSquaredNode;
    cudaKernelNodeParams bSquaredParams = {0};
    bSquaredParams.blockDim = BLOCK_SIZE;
    bSquaredParams.gridDim = gridSize;
    bSquaredParams.func = (void *)elementwiseProduct;
    void *bSquaredfunc_params[4] = {(void *)&b, (void *)&b, (void *)&sol1, (void *) &NUM_ELEMENTS};
    bSquaredParams.kernelParams = (void **)bSquaredfunc_params;

    cudaGraphNode_t neg4acNode;
    cudaKernelNodeParams neg4acParams = {0};
    neg4acParams.blockDim = BLOCK_SIZE;
    neg4acParams.gridDim = gridSize;
    neg4acParams.func = (void *)elementScalarProduct;
    const float NEG_FOUR = -4.0;
    void *neg4acfunc_params[5] = {(void *)&a, (void *)&c, (void *)&NEG_FOUR, (void *)&sol2, (void *) &NUM_ELEMENTS};
    neg4acParams.kernelParams = (void **)neg4acfunc_params;

    cudaGraphNode_t determinantSumNode;
    cudaKernelNodeParams determinantSumParams = {0};
    determinantSumParams.blockDim = BLOCK_SIZE;
    determinantSumParams.gridDim = gridSize;
    determinantSumParams.func = (void *)elementwiseSum;
    void *determinantSumfunc_params[4] = {(void *)&sol1, (void *)&sol2, (void *)&sol1, (void *) &NUM_ELEMENTS};
    determinantSumParams.kernelParams = (void **)determinantSumfunc_params;

    cudaGraphNode_t determinantSqrtNode;
    cudaKernelNodeParams determinantSqrtParams = {0};
    determinantSqrtParams.blockDim = BLOCK_SIZE;
    determinantSqrtParams.gridDim = gridSize;
    determinantSqrtParams.func = (void *)elementwiseSqrt;
    void *determinantSqrtfunc_params[3] = {(void *)&sol1, (void *)&tmp, (void *) &NUM_ELEMENTS};
    determinantSqrtParams.kernelParams = (void **)determinantSqrtfunc_params;
    
    cudaGraphNode_t bPlusNode;
    cudaKernelNodeParams bPlusParams = {0};
    bPlusParams.blockDim = BLOCK_SIZE;
    bPlusParams.gridDim = gridSize;
    bPlusParams.func = (void *)elementwiseSum;
    void *bPlusfunc_params[4] = {(void *)&tmp, (void *)&b, (void *)&sol1, (void *) &NUM_ELEMENTS};
    bPlusParams.kernelParams = (void **)bPlusfunc_params;
    
    cudaGraphNode_t bMinusNode;
    cudaKernelNodeParams bMinusParams = {0};
    bMinusParams.blockDim = BLOCK_SIZE;
    bMinusParams.gridDim = gridSize;
    bMinusParams.func = (void *)elementwiseDifference;
    void *bMinusfunc_params[4] = {(void *)&b, (void *)&tmp,  (void *)&sol2, (void *) &NUM_ELEMENTS};
    bMinusParams.kernelParams = (void **)bMinusfunc_params;

    cudaGraphNode_t bPlusQuotientNode;
    cudaKernelNodeParams bPlusQuotientParams = {0};
    bPlusQuotientParams.blockDim = BLOCK_SIZE;
    bPlusQuotientParams.gridDim = gridSize;
    bPlusQuotientParams.func = (void *)elementwiseQuotient;
    const float ONE_HALF = 0.5;
    void *bPlusQuotientfunc_params[5] = {(void *)&sol1, (void *)&a, (void *)&ONE_HALF, (void *)&sol1, (void *) &NUM_ELEMENTS};
    bPlusQuotientParams.kernelParams = (void **)bPlusQuotientfunc_params;

    cudaGraphNode_t bMinusQuotientNode;
    cudaKernelNodeParams bMinusQuotientParams = {0};
    bMinusQuotientParams.blockDim = BLOCK_SIZE;
    bMinusQuotientParams.gridDim = gridSize;
    bMinusQuotientParams.func = (void *)elementwiseQuotient;
    void *bMinusQuotientfunc_params[5] = {(void *)&sol2, (void *)&a, (void *)&ONE_HALF, (void *)&sol2, (void *) &NUM_ELEMENTS};
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
    
    cudaGraphExec_t graphExecutable;
    cudaGraphInstantiate(&graphExecutable, quadraticGraph, NULL, NULL, 0);


    cudaGraphLaunch(graphExecutable, 0);

    cudaStreamSynchronize(0);

    cudaGraphDestroy(quadraticGraph);
    cudaGraphExecDestroy(graphExecutable);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(tmp);
    cudaFree(sol1);
    cudaFree(sol2);
}