#include "utils.cuh"
#include "kernels.cuh"

void random_mat(float * A, int M, int N) {
	srand(time(NULL));
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			A[i * N + j] = (rand() % 10) + (float)rand()/(float)RAND_MAX;
		}
	}
}

void compare_mat(float *A, float *B, int M, int N) {
	float max_err = 0;
	float ave_err = 0;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			if (B[i*N + j]!=0) {
				float err = fabs((A[i*N + j] - B[i*N + j])/B[i*N + j]);
				if (err > max_err) max_err = err;
				ave_err += err;
			}
		}
	}
	ave_err /= M * N;
	printf("max error: %g, average error: %g\n", max_err, ave_err);
}

void copy_mat(float *dst, const float *src, int m, int n) {
	for (int i = 0; i < m*n; i++) {
		*(dst + i) = *(src + i);
	}
}

void print_mat(const float *A, int M, int N, int m, int n) {
	for (int i = 0; i < MIN(M, m); i++) {
		for (int j = 0; j < MIN(N, n); j++) {
			printf("%.3f ",A[i*N + j]);
		}
		printf("\n");
	}
}

clock_t cpu_matmult(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
	clock_t start = clock();
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			double tmp = 0;
			for (int k = 0; k < K; k++) {
				tmp += A[i*K + k] * B[k*N + j]; 
			}
			C[i*N + j] = alpha * tmp + beta * C[i*N + j];
		}
	}
	clock_t end = clock();
	return end - start;
}

int init_cuda() {
	int count;
	cudaGetDeviceCount(&count);
	if (count == 0) {
		fprintf(stderr, "There is no device\n");
		return -1;
	}
	printf("There are %d device.\n", count);
	int i;
	for (i = 0; i < count; i++) {
		struct cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) break;
		}
	}
	if (i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return -1;
	}
	cudaSetDevice(i);
	return 0;
}

void CudaDeviceInfo() {
    int deviceId;

    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    /*
   * There should be no need to modify the output string below.
   */

    printf("Device ID: %d\n\
       *Number of SMs: %d\n\
       Compute Capability Major: %d\n\
       Compute Capability Minor: %d\n\
       memoryBusWidth: %d\n\
       *maxThreadsPerBlock: %d\n\
       maxThreadsPerMultiProcessor: %d\n\
       *totalGlobalMem: %zuM\n\
       sharedMemPerBlock: %zuKB\n\
       *sharedMemPerMultiprocessor: %zuKB\n\
       totalConstMem: %zuKB\n\
       *multiProcessorCount: %d\n\
       *Warp Size: %d\n",
           deviceId,
           props.multiProcessorCount,
           props.major,
           props.minor,
           props.memoryBusWidth,
           props.maxThreadsPerBlock,
           props.maxThreadsPerMultiProcessor,
           props.totalGlobalMem / 1024 / 1024,
           props.sharedMemPerBlock / 1024,
           props.sharedMemPerMultiprocessor / 1024,
           props.totalConstMem / 1024,
           props.multiProcessorCount,
           props.warpSize);
};


void testkernel_v1(float *A, float *B,  float *C, int M, int N, int K, float alpha, float beta) {
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(DIV_CEL(N,BLOCK_SIZE), DIV_CEL(M, BLOCK_SIZE));
	sgemm_naive<<<grid,block>>>(A, B, C, M, N, K, alpha, beta);
}

void testkernel_v2(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
	dim3 block(BLOCK_SIZE * BLOCK_SIZE);
	dim3 grid(DIV_CEL(N,BLOCK_SIZE), DIV_CEL(M, BLOCK_SIZE));
	sgemm_global_mem_coalesce<BLOCK_SIZE> <<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
}

void testkernel_v3(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
	dim3 block(BLOCK_SIZE * BLOCK_SIZE);
	dim3 grid(DIV_CEL(N,BLOCK_SIZE), DIV_CEL(M, BLOCK_SIZE));
	sgemm_shared_mem<BLOCK_SIZE> <<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
}
void testkernel(int kernel_num, float *A, float *B, float *C, int M, int N, 
		int K, float alpha, float beta) {
	switch (kernel_num) {
		case 1:
			testkernel_v1(A, B, C, M, N, K, alpha, beta);
			break;
		case 2:
			testkernel_v2(A, B, C, M, N, K, alpha, beta);
			break;
		case 3:
			testkernel_v3(A, B, C, M, N, K, alpha, beta);
			break;
			
		default:
			break;

	}
}
