#include <cstdio>

#include "repeat.h"

__global__ void timeSharedCopy(int *input, int size, int *durations, int iterations) {
	__shared__ extern int shared_data[];
	unsigned int j;

	for (int i=0; i<size; i++)
		shared_data[i] = input[i];

	int clockStart, clockStop;

	for (int i=0; i<iterations; i++) {
		clockStart = clock();
		j = shared_data[0];
		clockStop = clock();

		durations[i] = clockStop - clockStart;
	}

	input[size - 1] = j;
}

int main(int argc, char *argv[])
{
	const int NUM_ELEMENTS = 64;
	const int ITERATIONS = 16;

	int h_input[NUM_ELEMENTS] = {0};
	int h_durations[ITERATIONS] = {0};
	
	int *d_input, *d_durations;
	cudaMalloc(&d_input, sizeof(int)*NUM_ELEMENTS);
	cudaMalloc(&d_durations, sizeof(int)*ITERATIONS);

	cudaMemcpy(d_input, h_input, sizeof(int)*NUM_ELEMENTS, cudaMemcpyHostToDevice);

	timeSharedCopy<<<1, 1, sizeof(int)*NUM_ELEMENTS>>>(d_input, NUM_ELEMENTS, d_durations, ITERATIONS);

	cudaMemcpy(h_durations, d_durations, sizeof(int)*ITERATIONS, cudaMemcpyDeviceToHost);

	for (int i=0; i<ITERATIONS; i++)
		printf("%d ",*h_durations);
	printf("\n");

	cudaFree(d_input);
	cudaFree(d_durations);
}
