#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "kernels.cuh"

#define MAX(a, b) (((a) > (b))?(a):(b))
#define MIN(a, b) (((a) < (b))?(a):(b))

#define DIV_CEL(M,N) ((M + N - 1)/N)
#define BLOCK_SIZE 32

void random_mat(float * a, int m, int n);

void compare_mat(float *A, float *B, int M, int N);

void copy_mat(float *dst, const float *src, int m, int n);
clock_t cpu_matmult(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);
void print_mat(const float *A, int M, int N, int m, int n);
int init_cuda();
void CudaDeviceInfo();

void testkernel_v1(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);
void testkernel(int kernel_num, float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);
