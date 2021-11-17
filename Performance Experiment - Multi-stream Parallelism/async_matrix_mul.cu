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

int main(int argc, char *arg[]) {
    // Number of rows in output matrix
    const int ROWS = 32;

    // Number of columns in output matrix
    const int COLS = 16;

    // Length of aligned dimension in input matrices
    const int ALIGNED = 131072;

    // Length of each block axis - blocks contain 256 threads
    const int BLOCK_LENGTH = 16;

    // The max number of streams to run at once - and the number of matrices to multiply
    const int MAX_STREAMS = 16;

    const int NUM_TRIALS = 20;

    dim3 grid_A = dim3(ALIGNED / BLOCK_LENGTH + 1, ROWS / BLOCK_LENGTH + 1);
    dim3 grid_B = dim3(COLS / BLOCK_LENGTH + 1, ALIGNED / BLOCK_LENGTH + 1);

    dim3 blockSize = dim3(BLOCK_LENGTH, BLOCK_LENGTH);

    // Initialize and fill two input and one output matrix per multiplication problem
    float *d_A[MAX_STREAMS], *d_B[MAX_STREAMS], *d_C[MAX_STREAMS];
    for (int i=0; i<MAX_STREAMS; i++) {
        cudaMalloc(&(d_A[i]), sizeof(float)*ROWS*ALIGNED);
        cudaMalloc(&(d_B[i]), sizeof(float)*ALIGNED*COLS);
        cudaMalloc(&(d_C[i]), sizeof(float)*ROWS*COLS);

        matrix_fill<<<grid_A, blockSize>>>(d_A[i], ROWS, ALIGNED, 1);
        matrix_fill<<<grid_B, blockSize>>>(d_B[i], ALIGNED, COLS, 1);
    }

    // Stream pool to be used in all timing tests
    cudaStream_t streams[MAX_STREAMS];
    for (int s=0; s<MAX_STREAMS; s++)
        cudaStreamCreate(&streams[s]);

    cudaEvent_t clockStart, clockStop;
    cudaEventCreate(&clockStart);
    cudaEventCreate(&clockStop);

    // Print CSV-style header row
    for (int i=0; i<MAX_STREAMS; i++) {
        printf("%d Stream",i+1);
        if (i>0)
            printf("s");
        // Newline after last item
        if (MAX_STREAMS-i==1)
            printf("\n");
        // Otherwise comma
        else
            printf(",");
    }

    // Perform each set of tests NUM_TRIALS times so we can get an average measure of performance
    for (int trial=0; trial<NUM_TRIALS; trial++){

        // Iterate through number of streams used to observe performance impact
        for (int streamsUsed = 1; streamsUsed <= MAX_STREAMS; streamsUsed++) {

            dim3 gridSize = dim3(COLS / BLOCK_LENGTH, ROWS / BLOCK_LENGTH);
            gridSize.x += (COLS % BLOCK_LENGTH) ? 1 : 0;
            gridSize.y += (ROWS % BLOCK_LENGTH) ? 1 : 0;

            cudaEventRecord(clockStart, 0);

                for (int i=0; i<MAX_STREAMS; i++)
                    // Split problems between active streams -> streams[i%streamsUsed]
                    matrix_mul<<<gridSize, blockSize, 0, streams[i%streamsUsed]>>>(d_A[i], d_B[i], d_C[i], ROWS, COLS, ALIGNED);

                for (int s=0; s<streamsUsed; s++)
                    cudaStreamSynchronize(streams[s]);

            cudaEventRecord(clockStop, 0);

            float timeElapsed;
            cudaEventSynchronize(clockStop);
            cudaEventElapsedTime(&timeElapsed, clockStart, clockStop);

            // Print results in CSV format
            printf("%.4f", timeElapsed);
            if (streamsUsed==MAX_STREAMS)
                printf("\n");
            else
                printf(",");
        }
    }

    // Cleanup
    cudaEventDestroy(clockStart);
    cudaEventDestroy(clockStop);

    for (int s=0; s<MAX_STREAMS; s++)
        cudaStreamDestroy(streams[s]);

    for (int i=0; i<8; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
    }
}
