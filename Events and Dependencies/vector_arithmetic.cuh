// Calculates unique thread index for 1, 2, and 3 dimensional kernels
__device__ int globalIndex() {
    // Amount of threads in a 1-block deep slice of the grid
    int threadsPerGridSlice = (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y) * blockDim.z;

    // Amount of threads in a 1-thread deep slice of a block
    int threadsPerBlockSlice = (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y);

    // Amount of threads per row of blocks across the grid
    int threadsPerBlockRow = gridDim.x * blockDim.x * blockDim.y;

    // Amount of threads per row of threads across the grid
    int threadsPerThreadRow = blockDim.x * gridDim.x;

                    //First account for all whole blocks in the z-dimension
    int threadID = threadsPerGridSlice * blockIdx.z;
    // Then count all threads in prior z-slices in the same block
    threadID += threadsPerBlockSlice * threadIdx.z;
    // Then count all the threads in prior grid rows
    threadID += threadsPerBlockRow * blockIdx.y;
    // Then count all threads in prior rows within the same block
    threadID += threadsPerThreadRow * threadIdx.y;
    // Then account for threads in blocks in the same row as current block
    threadID += blockDim.x * blockIdx.x;
    // Then add rows in the same row, z-slice, and block before this thread 
    threadID += threadIdx.x;

    return threadID;
}

__global__ void fillArray(float *array, int numElements) {
    int id = globalIndex();

    if (id < numElements)
        array[id] = id;
}

__global__ void fillArray(float *array, int numElements, int value) {
    int id = globalIndex();

    if (id < numElements)
        array[id] = value;
}

// Multiply each element in a[] by scalar
__global__ void scalarProduct(float *a, float scalar, float *out, float numElements) {
    int id = globalIndex();

    if (id < numElements)
        out[id] = a[id] * scalar;
}

__global__ void elementwiseProduct(float *a, float *b, float *out, int numElements) {
    int id = globalIndex();

    if (id < numElements)
        out[id] = a[id] * b[id];
}

__global__ void elementScalarProduct(float *a, float *b, float scalar, float *out, int numElements) {
    int id = globalIndex();

    if (id < numElements)
        out[id] = a[id] * b[id] * scalar;
}

__global__ void elementwiseQuotient(float *numerator, float *denominator, float scalar, float *out, int numElements) {
    int id = globalIndex();

    if (id < numElements) {
        if (denominator == 0)
            out[id] = 0;
        else
            out[id] = scalar * (numerator[id] /  denominator[id]);
    }
}

__global__ void elementwiseSum(float *a, float *b, float *out, int numElements) {
    int id = globalIndex();

    if (id < numElements)
        out[id] = a[id] + b[id];
}

__global__ void elementwiseDifference(float *minuend, float *subtrahend, float *difference, int numElements) {
    int id = globalIndex();

    if (id < numElements)
        difference[id] = minuend[id] - subtrahend[id];
}

__global__ void elementwiseSqrt(float *a, float *out, int numElements) {
    int id = globalIndex();

    if (id < numElements) {
        if (a[id] <= 0)
            out[id] = 0;
        else
            out[id] = sqrtf(a[id]);
    }
}