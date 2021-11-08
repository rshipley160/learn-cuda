#include <cstdio>

// Fill matrix - matrix size is determined by kernel configuration
__global__ void matrix_fill(float *matrix) {
    // The row of the output cell
    int i = blockDim.y * blockIdx.y + threadIdx.y;

    // The column of the output cell
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    // Number of threads/elements per grid row
    int rowSize = gridDim.x * blockDim.x;

    matrix[rowSize*i + j] = j; // Replace j with whatever value you want to set
}

__global__ void matrix_mul(float *A, float *B, float *C, int hiddenDim) {
    // The row of the output cell
    int i = blockDim.y * blockIdx.y + threadIdx.y;

    // The column of the output cell
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    // Total number of threads in one row of the kernel grid
    int rowSize = gridDim.x * blockDim.x;

    for (int k = 0; k <= hiddenDim; k++) {
        C[rowSize*i + j] += A[rowSize*i + k] * B[rowSize*k + j];
    }
}



int main(int argc, char *arg[]) {
    // Number of rows in output matrix
    const int ROWS = 8;

    // Number of columns in output matrix
    const int COLS = 8;

    // Length of aligned dimension in input matrices
    const int ALIGNED = 8;

    // Device input and output arrays (matrices)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float)*ROWS*ALIGNED);
    cudaMalloc(&d_B, sizeof(float)*ALIGNED*COLS);
    cudaMalloc(&d_C, sizeof(float)*ROWS*COLS);

    // Host copy of output so we can print the result
    float *h_C;
    h_C = (float *) malloc(sizeof(float)*ROWS*COLS);

    matrix_fill<<<1, dim3(ROWS, ALIGNED)>>>(d_A);
    matrix_fill<<<1, dim3(ALIGNED, COLS)>>>(d_B);

    matrix_mul<<<1, dim3(ROWS, COLS)>>>(d_A, d_B, d_C, ALIGNED);

    cudaMemcpy(h_C, d_C, sizeof(float)*ROWS*COLS, cudaMemcpyDeviceToHost);

    for (int i=0; i<ROWS; i++) {
        for (int j=0; j<COLS; j++) {
            printf("%12f ",h_C[COLS*i + j]);
        }
        printf("\n");
    }
}
