#include <stdio.h>
#include "utils.cuh"

// M*K K*N 
// c = alpha*a@b + beta*c

int main() {
	if (init_cuda() == 0) {
		printf("CUDA initialized.\n");
	}
	else {
		printf("initialized CUDA fail!\n");
		return -1;
	}

	CudaDeviceInfo();

	float *A, *B, *C, *dA, *dB, *dC, *D;
	int M = 1000, N = 1000, K = 1000;
	float alpha = 1.0, beta = 0.0;
	A = (float *)malloc(sizeof(float)*M*K);
	B = (float *)malloc(sizeof(float)*K*N);
	C = (float *)malloc(sizeof(float)*M*N);
	D = (float *)malloc(sizeof(float)*M*N);

	random_mat(A, M, K);
	random_mat(B, K, N);
	random_mat(C, M, N);

	cudaMalloc((void **)&dA, sizeof(float)*M*K);
	cudaMalloc((void **)&dB, sizeof(float)*K*N);
	cudaMalloc((void **)&dC, sizeof(float)*M*N);

	cudaMemcpy(dA, A, sizeof(float)*M*K, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeof(float)*K*N, cudaMemcpyHostToDevice);
	cudaMemcpy(dC, C, sizeof(float)*M*N, cudaMemcpyHostToDevice);

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start, 0);

	testkernel(1, dA, dB, dC, M, N, K, alpha, beta);

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);

	cudaMemcpy(D, dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

	float msec = 0;
	cudaEventElapsedTime(&msec, start, end);
	float sec = msec/1000;
	printf("(GPU)Time used: %.4f sec(%.2lf GFLOPS)\n", sec,
			2.0*M*K*N/(sec*1E9));

	clock_t cpu_time = cpu_matmult(A, B, C, M, N, K, alpha, beta);
	sec = cpu_time/(float)CLOCKS_PER_SEC;
	printf("(CPU)Time used: %.4f sec(%.2lf GFLOPS)\n", sec,
			2.0*M*K*N/(sec*1E9));

	compare_mat(C, D, M, N);
	// int x = MIN(M,3);
	// int y = MIN(N,3);
	// printf("x = %d, y = %d\n", x, y);
	print_mat(C, M, N, 3, 3);
	print_mat(D, M, N, 3, 3);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
	cudaFree(start);
	cudaFree(end);
}

