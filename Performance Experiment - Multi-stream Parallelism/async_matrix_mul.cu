#include <cstdio>

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

__global__ void matrix_mul_shared(float *A, float *B, float *C, int rows, int cols, int aligned, int streamID, int numStreams) {
    // The row of the output cell
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    // Cut out any threads that extend past the output matrix
    if (i >= rows) return;

    // The column of the output cell
    int j = streamID * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;
    // Cut out any threads that extend past the output matrix
    if (j >= cols) return;

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

int main(int argc, char *arg[]) {
    // Number of rows in output matrix
    const int ROWS = 4;

    // Number of columns in output matrix
    const int COLS = 64;

    // Length of aligned dimension in input matrices
    const int ALIGNED = 16382;

    const int BLOCK_LENGTH = 4;

    // Device input and output arrays (matrices)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float)*ROWS*ALIGNED);
    cudaMalloc(&d_B, sizeof(float)*ALIGNED*COLS);
    cudaMalloc(&d_C, sizeof(float)*ROWS*COLS);


    dim3 grid_A = dim3(ALIGNED / BLOCK_LENGTH + 1, ROWS / BLOCK_LENGTH + 1);
    dim3 grid_B = dim3(COLS / BLOCK_LENGTH + 1, ALIGNED / BLOCK_LENGTH + 1);

    dim3 blockSize = dim3(BLOCK_LENGTH, BLOCK_LENGTH);

    matrix_fill<<<grid_A, blockSize>>>(d_A, ROWS, ALIGNED, ALIGNED);
    matrix_fill<<<grid_B, blockSize>>>(d_B, ALIGNED, COLS, ALIGNED);

    // printf("gridA(%d,%d); gridB(%d,%d)\n",grid_A.x, grid_A.y, grid_B.x, grid_B.y);
    // printf("%s\n",cudaGetErrorString(cudaGetLastError()));
    // printMatrix(d_A, ROWS, ALIGNED);
    // printMatrix(d_B, ALIGNED, COLS);

    cudaEvent_t clockStart, clockStop;
    cudaEventCreate(&clockStart);
    cudaEventCreate(&clockStop);

    for (int numStreams = 1; numStreams <= 8; numStreams++) {

        dim3 gridSize = dim3(COLS / numStreams / BLOCK_LENGTH, ROWS / BLOCK_LENGTH);
        gridSize.x += (COLS % (numStreams*BLOCK_LENGTH)) ? 1 : 0;
        gridSize.y += (ROWS % BLOCK_LENGTH) ? 1 : 0;

        cudaStream_t streams[numStreams];
        for (int s=0; s<numStreams; s++) {
            cudaStreamCreate(&(streams[s]));
        }

        cudaEventRecord(clockStart, 0);

        for (int s=0; s<numStreams; s++) {
            matrix_mul_shared<<<gridSize, blockSize, 0, streams[s]>>>(d_A, d_B, d_C, ROWS, COLS, ALIGNED, s, numStreams);    
        }

        for (int s=0; s<numStreams; s++) {
            cudaStreamSynchronize(streams[s]);
        }

        cudaEventRecord(clockStop, 0);
        float timeElapsed;
        cudaEventSynchronize(clockStop);
        cudaEventElapsedTime(&timeElapsed, clockStart, clockStop);

            printf("%d stream(s): %.4f\n", numStreams, timeElapsed);

        for (int s=0; s<numStreams; s++){
            cudaStreamDestroy(streams[s]);
        }
    }
    cudaEventDestroy(clockStart);
    cudaEventDestroy(clockStop);
}
