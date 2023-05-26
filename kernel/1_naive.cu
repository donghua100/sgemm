#include<cuda_runtime.h>

__global__ void sgemm_naive(const float *A, const float *B,
		float *C, int M, int N, int K, float alpha, float beta) {
	int x =  blockIdx.x*blockDim.x + threadIdx.x;
	int y =  blockIdx.y*blockDim.y + threadIdx.y;

	if (x < M && y < N) {
		float tmp = 0;
		for (int i = 0; i < K; i++) {
			tmp += A[x*K + i] * B[i*N + y];
		}
		C[x*N + y] = alpha*tmp + beta*C[x*N + y];
	}
}



