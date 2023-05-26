#pragma once
__global__ void sgemm_naive(const float *A, const float *B,
		float *C, int M, int N, int K, float alpha, float beta);

