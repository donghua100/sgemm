#pragma once
#include "1_naive.cuh"
#include "2_global_mem_coalesce.cuh"



// __global__ void sgemm_naive(const float *A, const float *B,
// 		float *C, int M, int N, int K, float alpha, float beta);
//
// // template<const int BLOCKSIZE>
// __global__ void sgemm_global_mem_coalesce(const float *A, const float *B,
// 		float * C, int M, int N, int K, float alpha, float beta);
