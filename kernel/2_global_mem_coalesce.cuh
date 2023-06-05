#pragma once

#include <cuda_runtime.h>


template<const int BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(const float *A, const float *B,
		float * C, int M, int N, int K, float alpha, float beta) {
	int x = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
	int y = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
	if ( x < N && y < M) {
		float tmp = 0;
		for (int i = 0; i < K; i++) {
			tmp += A[y*K + i] * B[i*N + x];
		}
		C[y*N + x] = alpha*tmp + beta*C[y*N + x];
	}
} 
