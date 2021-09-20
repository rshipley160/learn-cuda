__global__ void fillWithOnes(int *array, int size) {
    if (threadIdx.x < size)
        array[threadIdx.x] = 1;
}

int main(int argc, char *argv[]) {
    int arraySize = 256;
    int *d_array;

    cudaMalloc(&d_array, sizeof(int)*arraySize);

    fillWithOnes<<<1, 256>>>(d_array, arraySize);
}
