#include <cstdio>
#define BLOCK_LENGTH 8
#define BLOCK_DIM dim3(BLOCK_LENGTH, BLOCK_LENGTH)

// Fill matrix of given size with value
__global__ void matrix_fill(float *matrix, int rows, int cols, float value) {
    // The row of the output cell
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= rows) return;

    // The column of the output cell
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j >= cols) return;

    matrix[cols*i + j] = value;
}

__global__ void matrix_mul(float *A, float *B, float *C, int rows, int cols, int aligned) {
    // The row of the output cell
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= rows) return;

    // The column of the output cell
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j >= cols) return;

    // Clear any previous values in output matrix
    C[cols*i + j] = 0;

    for (int k = 0; k < aligned; k++) {
        C[cols*i + j] += A[aligned*i + k] * B[cols*k + j];
    }
}

void printMatrix(float *d_A, int rows, int cols) {
    float *h_A;
    h_A = (float *) malloc(sizeof(float)*rows*cols);

    cudaMemcpy(h_A, d_A, sizeof(float)*rows*cols, cudaMemcpyDeviceToHost);

    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            printf("%3.0f ",h_A[cols*i + j]);
        }
        printf("\n");
    }
    printf("\n");

    free(h_A);
}

float syncDoubleMatrix(int rows, int cols, int vectorLength) {
    float *A, *B, *C; // A and B are inputs, C is output
    cudaMalloc(&A, sizeof(float)*rows*vectorLength);
    cudaMalloc(&B, sizeof(float)*cols*vectorLength);
    cudaMalloc(&C, sizeof(float)*rows*cols);

    float *A2, *B2, *C2; // A2 and B2 are inputs, C2 is output
    cudaMalloc(&A2, sizeof(float)*rows*vectorLength);
    cudaMalloc(&B2, sizeof(float)*cols*vectorLength);
    cudaMalloc(&C2, sizeof(float)*rows*cols);

    cudaEvent_t clockStart, clockStop;
    cudaEventCreate(&clockStart);
    cudaEventCreate(&clockStop);

    // Fill input matrices with 1s
    matrix_fill<<<dim3(vectorLength / BLOCK_LENGTH + 1, rows / BLOCK_LENGTH + 1), BLOCK_DIM>>>(A, rows, vectorLength, 1);
    matrix_fill<<<dim3(cols/ BLOCK_LENGTH + 1, vectorLength / BLOCK_LENGTH + 1), BLOCK_DIM>>>(B, vectorLength, cols, 1);

    matrix_fill<<<dim3(vectorLength / BLOCK_LENGTH + 1, rows / BLOCK_LENGTH + 1), BLOCK_DIM>>>(A2, rows, vectorLength, 1);
    matrix_fill<<<dim3(cols/ BLOCK_LENGTH + 1, vectorLength / BLOCK_LENGTH + 1), BLOCK_DIM>>>(B2, vectorLength, cols, 1);

    // Do a dry run to warm up the GPU
    matrix_mul<<<dim3(cols / BLOCK_LENGTH + 1, rows / BLOCK_LENGTH + 1), BLOCK_DIM>>>(A, B, C, rows, cols, vectorLength);
    matrix_mul<<<dim3(cols / BLOCK_LENGTH + 1, rows / BLOCK_LENGTH + 1), BLOCK_DIM>>>(A2, B2, C2, rows, cols, vectorLength);

    // Start the timer and do it for real    
    cudaEventRecord(clockStart);
    
       matrix_mul<<<dim3(cols / BLOCK_LENGTH + 1, rows / BLOCK_LENGTH + 1), BLOCK_DIM>>>(A2, B2, C2, rows, cols, vectorLength);
       matrix_mul<<<dim3(cols / BLOCK_LENGTH + 1, rows / BLOCK_LENGTH + 1), BLOCK_DIM>>>(A, B, C, rows, cols, vectorLength);

    cudaEventRecord(clockStop);
    cudaEventSynchronize(clockStop);

    float timeElapsed;
    cudaEventElapsedTime(&timeElapsed, clockStart, clockStop);

    cudaEventDestroy(clockStart);
    cudaEventDestroy(clockStop);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    cudaFree(A2);
    cudaFree(B2);
    cudaFree(C2);

    return timeElapsed;
}

float asyncDoubleMatrix(int rows, int cols, int vectorLength) {
    float *A, *B, *C; // A and B are inputs, C is output
    cudaMalloc(&A, sizeof(float)*rows*vectorLength);
    cudaMalloc(&B, sizeof(float)*cols*vectorLength);
    cudaMalloc(&C, sizeof(float)*rows*cols);

    float *A2, *B2, *C2; // A2 and B2 are inputs, C2 is output
    cudaMalloc(&A2, sizeof(float)*rows*vectorLength);
    cudaMalloc(&B2, sizeof(float)*cols*vectorLength);
    cudaMalloc(&C2, sizeof(float)*rows*cols);

    cudaStream_t secondStream;
    //cudaStreamCreate(&secondStream); - blocks on default stream
    cudaStreamCreateWithFlags(&secondStream, cudaStreamNonBlocking); //Always runs asynchronously

    cudaEvent_t clockStart, clockStop;
    cudaEventCreate(&clockStart);
    cudaEventCreate(&clockStop);

    // Fill input matrices with 1s
    matrix_fill<<<dim3(vectorLength / BLOCK_LENGTH + 1, rows / BLOCK_LENGTH + 1), BLOCK_DIM>>>(A, rows, vectorLength, 1);
    matrix_fill<<<dim3(cols/ BLOCK_LENGTH + 1, vectorLength / BLOCK_LENGTH + 1), BLOCK_DIM>>>(B, vectorLength, cols, 1);

    matrix_fill<<<dim3(vectorLength / BLOCK_LENGTH + 1, rows / BLOCK_LENGTH + 1), BLOCK_DIM>>>(A2, rows, vectorLength, 1);
    matrix_fill<<<dim3(cols/ BLOCK_LENGTH + 1, vectorLength / BLOCK_LENGTH + 1), BLOCK_DIM>>>(B2, vectorLength, cols, 1);

    // Do a dry run to warm up the GPU
    matrix_mul<<<dim3(cols / BLOCK_LENGTH + 1, rows / BLOCK_LENGTH + 1), BLOCK_DIM>>>(A, B, C, rows, cols, vectorLength);
    matrix_mul<<<dim3(cols / BLOCK_LENGTH + 1, rows / BLOCK_LENGTH + 1), BLOCK_DIM, 0, secondStream>>>(A2, B2, C2, rows, cols, vectorLength);

    // Wait on previous async operations to complete before starting the timer
    cudaStreamSynchronize(secondStream);

    // Start the timer and do it for real    
    cudaEventRecord(clockStart);
    
        matrix_mul<<<dim3(cols / BLOCK_LENGTH + 1, rows / BLOCK_LENGTH + 1), BLOCK_DIM>>>(A, B, C, rows, cols, vectorLength);
        matrix_mul<<<dim3(cols / BLOCK_LENGTH + 1, rows / BLOCK_LENGTH + 1), BLOCK_DIM, 0, secondStream>>>(A2, B2, C2, rows, cols, vectorLength);
       
       // Wait on timed operations to complete before stopping the timer
       cudaStreamSynchronize(secondStream);

    cudaEventRecord(clockStop);
    cudaEventSynchronize(clockStop);

    float timeElapsed;
    cudaEventElapsedTime(&timeElapsed, clockStart, clockStop);

    cudaEventDestroy(clockStart);
    cudaEventDestroy(clockStop);

    cudaStreamDestroy(secondStream);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    cudaFree(A2);
    cudaFree(B2);
    cudaFree(C2);

    return timeElapsed;
}

int main(int argc, char *argv[]) {

    //Size of output matrix - # of rows in input A, # cols in B
    int ROWS = 64;
    int COLS = 64;

    // Other dimension of inputs - cols of A, rows of B
    const int VECTOR_LENGTH = 131072;

    float syncCompletionTime = syncDoubleMatrix(ROWS, COLS, VECTOR_LENGTH);

    printf("Synchronous completion time: %f\n", syncCompletionTime);

    float asyncCompletionTime = asyncDoubleMatrix(ROWS, COLS, VECTOR_LENGTH);

    printf("Asynchronous completion time: %f\n", asyncCompletionTime);

}