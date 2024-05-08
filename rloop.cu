#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <emmintrin.h>
#include <limits.h>
#include <pmmintrin.h>
#include <immintrin.h>
# include <math.h>
# include <omp.h>
# include <stdlib.h>
#include <emmintrin.h>
#include <stdint.h>	
#include <omp.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <time.h>
//#include <sched.h>
//#include <pthread.h>
//#include <sys/syscall.h>
//#include <sys/mman.h>
#include <omp.h>

//Macros
#define EPSILON 0.0001

const int N = 169;//5//80
void array_initialization_CPU();
void default_kernel();
unsigned short int equal(float const a, float const b);
void results_check();
int create_arrays_GPU();
//__global__ void kernel(float* sum_GPU, float* A_GPU, float* C_GPU);
__global__ void cuda_layer_v1(float* sum_GPU, float* A_GPU, float* C_GPU);

cudaError_t errorCuda;
__declspec(align(4)) float sum[N][N][N], A[N][N][N], C[N][N], sum_optimized[N][N][N], t[N];
bool b = false;
int  r, q, s, p;
float* psum[N];
float* pC[N];
float* pA;

float* sum_GPU;
float* A_GPU;
float* C_GPU;

cudaPitchedPtr pitchPtr;
size_t pitch;
//Pointers to the arrays on the CPU
float* N_GPU;
const int* pN = &N;


int main()
{
    __int64 total_floating_point_operations = 2 * N;
    total_floating_point_operations = total_floating_point_operations * N * N * N;
    double total_floating_point_operations_converted = total_floating_point_operations;

    float cuda_time;
    cudaEvent_t timer_start, timer_stop;
    cudaEventCreate(&timer_start);
    cudaEventCreate(&timer_stop);
    //CPU arrays' memory allocation and initializing. 
    array_initialization_CPU();

    //Run default kernel and get executon time.
    double start = omp_get_wtime();
    default_kernel();
    double end = omp_get_wtime();
    double result = end - start;
    printf("Default kernel time: %lf\n", result);

    //GPU arrays' memory allocation and initializing.
    create_arrays_GPU();

    //Starting CUDA timer
    cudaEventRecord(timer_start, 0);

    //Copying from CPU to GPU
    errorCuda = cudaMemcpy(C_GPU, C, pow(N, 2) * sizeof(float), cudaMemcpyHostToDevice);
    errorCuda = cudaMemcpy(A_GPU, A, pow(N, 3) * sizeof(float), cudaMemcpyHostToDevice);
    errorCuda = cudaMemcpy(sum_GPU, sum_optimized, pow(N, 3) * sizeof(float), cudaMemcpyHostToDevice);
    errorCuda = cudaMemcpy(N_GPU, pN, sizeof(int), cudaMemcpyHostToDevice);

    //CUDA kernel
    dim3 dimBlock(1024, 1, 1);
    dim3 dimGrid(65535, 1, 1);
    //kernel << <dimGrid, dimBlock> >> (sum_GPU, A_GPU, C_GPU);
    cuda_layer_v1 << <dimGrid, dimBlock >> > (sum_GPU, A_GPU, C_GPU);

    //Copying from GPU to CPU
    errorCuda = cudaMemcpy(sum_optimized, sum_GPU, unsigned long long(N * N) * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (errorCuda != cudaSuccess) {
        printf("\n Copying from GPU to CPU failed");
        return -1;
    }
    //Stopping CDA timer and getting result
    cudaEventRecord(timer_stop, 0);  //get timer value
    cudaEventSynchronize(timer_stop);
    cudaEventElapsedTime(&cuda_time, timer_start, timer_stop);
    cuda_time /= 1000; //convert to seconds


    //Calculating FLOPS for CUDA
    double mega_FLOPS_CUDA = total_floating_point_operations_converted / 1000000;
    mega_FLOPS_CUDA = mega_FLOPS_CUDA / cuda_time;

    //Calculating FLOPS for CPU
    double mega_FLOPS_CPU = total_floating_point_operations_converted / 1000000;
    mega_FLOPS_CPU = mega_FLOPS_CPU / result;
    //Comparing GPU and CPU result.
    results_check();

    printf("\nCUDA Mega FLOPS achieved = %lf", mega_FLOPS_CUDA);
    printf("\nCUDA time = %f", cuda_time);
    printf("\nCPU Mega FLOPS achieved = %lf", mega_FLOPS_CPU);
    printf("\nCPU time = %f", result);

    return 0;
}

void array_initialization_CPU() {

    float e = 0.12, p = 0.72;
    unsigned int i, j, k;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            C[i][j] = (j % 9) + p;
        }
    }

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                sum[i][j][k] = 0.0;
                A[i][j][k] = (((i + j) % 99) + e);
            }
        }
    }
}

//Doitgen in kernel

__global__ void cuda_layer_v1(float* sum_GPU, float* A_GPU, float* C_GPU)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    /*
        for (int r = 0; r < N; r++) {
            for (int q = 0; q < N; q++) {
                for (int s = 0; s < N; s++) {

                    for (int p = 0; p < N; p++) {
                        //sum_GPU[r][q][p] = sum_GPU[r][q][p] + A_GPU[r][q][s] * C_GPU[s][p];
                        sum_GPU[(r * N * N) + q * N + p] = sum_GPU[(r * N * N) + q * N + p] + A_GPU[(r * N * N) + q * N + s] * C_GPU[s * N + p];
                    }
            }
        }
    }*/

    if (q < (N * N)) {
        if (q < N) {
            int r = q;
            for (int s = 0; s < N; s++) {
                for (int p = 0; p < N; p++) {
                    //sum_GPU[r][q][p] = sum_GPU[r][q][p] + A_GPU[r][q][s] * C_GPU[s][p];
                    sum_GPU[(r * N * N) + 0 + p] = sum_GPU[(r * N * N) + 0 + p] + A_GPU[(r * N * N) + 0 + s] * C_GPU[s * N + p];
                }
            }
        }
        else {
            int increment = q / N;
            int r = q%N;
            int q1 = increment;
            for (int s = 0; s < N; s++) {
                for (int p = 0; p < N; p++) {
                    sum_GPU[(r * N * N) + q1 * N + p] = sum_GPU[(r * N * N) + q1 * N + p] + A_GPU[(r * N * N) + q1 * N + s] * C_GPU[s * N + p];
                }
            }
        }
    }

}



int create_arrays_GPU() {

    errorCuda = cudaMalloc((void**)&sum_GPU, pow(N, 3) * sizeof(float));//allocate memory dynamically 
    if (errorCuda != cudaSuccess) {//if the GPU memory asked is not available
        printf("\nCudaMalloc failed");
        cudaFree(sum_GPU);
        return -1;//returns unsuccessfully
    }

    errorCuda = cudaMalloc((void**)&A_GPU, pow(N, 3) * sizeof(float));//allocate memory dynamically 
    if (errorCuda != cudaSuccess) {//if the GPU memory asked is not available
        printf("\nCudaMalloc failed");
        cudaFree(A_GPU);
        return -1;//returns unsuccessfully
    }

    errorCuda = cudaMalloc((void**)&C_GPU, pow(N, 2) * sizeof(float));//allocate memory dynamically 
    if (errorCuda != cudaSuccess) {//if the GPU memory asked is not available
        printf("\nCudaMalloc failed");
        cudaFree(C_GPU);
        return -1;//returns unsuccessfully
    }
    return 0;
}

void default_kernel() {

    for (r = 0; r < N; r++) {
        for (q = 0; q < N; q++) {
            for (s = 0; s < N; s++) {
                for (p = 0; p < N; p++) {
                    sum[r][q][p] = sum[r][q][p] + A[r][q][s] * C[s][p];
                }
            }
        }
    }
}

void results_check() {

    for (r = 0; r < N; r++) {
        for (q = 0; q < N; q++) {
            for (s = 0; s < N; s++) {
                for (p = 0; p < N; p++) {
                    if (equal(sum[r][q][p], sum_optimized[r][q][p]) == 1) {
                        printf("\n wrong values: %f %f", sum[r][q][p], sum_optimized[r][q][p]);
                    }
                    else {
                      //  printf("\n correct: %f %f", sum[r][q][p], sum_optimized[r][q][p]);
                    }
                }
            }
        }
    }
}

unsigned short int equal(float const a, float const b) {
    float temp = a - b;

    if (b == 0.0f) {
        if (a == 0.0f) {
            return 0;
        }
        else {
            return 1;
        }
    }
    else {

        if ((fabs(temp) / fabs(b)) < EPSILON) {
            return 0;
        }
        else {
            return 1;
        }
    }
}