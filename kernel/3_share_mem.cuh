#include <cuda_runtime.h>

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem(const float *A, const float *B, float *C,
		int M, int N, int K, float alpha, float beta) {
	int bx = blockIdx.x;
	int by = blockIdx.y;

	const int BM = BLOCKSIZE;
	const int BK = BLOCKSIZE;
	const int BN = BLOCKSIZE;

	__shared__ float As[BM*BK];
	__shared__ float Bs[BK*BN];
	int tx = threadIdx.x % BLOCKSIZE;
	int ty = threadIdx.x / BLOCKSIZE;

	A += by * BLOCKSIZE * K;
	B += bx * BLOCKSIZE;
	C += by * BLOCKSIZE * K + bx * BLOCKSIZE;

	float tmp = 0;
	for (int k = 0; k < K; k += BK) {
		As[ty*BK + tx] = A[ty*K + tx];
		Bs[ty*BK + tx] = B[ty*N + tx];
		__syncthreads();
		A += BK;
		B += BK*N;
		for (int i = 0; i < BK; i++) {
			tmp += As[ty*BK + i] * Bs[i*BK + tx];
		}
		__syncthreads();
	}
	C[ty*N + tx] = alpha*tmp + beta*C[ty*N + tx];
}
