#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>

// nvcc -O3 -arch=sm_86 -o sigmoid_harness_naive sigmoid_harness_naive.cu
// 268435456

/// NAIVE
/*
tanh benchmark - N = 268435456 elements (1073.74 MB)
Average kernel time: 12.687 ms
Effective bandwidth: 253.90 GB/s
Correctness: PASSED
*/
/// NAIVE
/// V1
/*
tanh benchmark - N = 268435456 elements (1073.74 MB)
Average kernel time: 13.088 ms
Effective bandwidth: 246.13 GB/s
Correctness: PASSED
*/
/// V1
/// V2
/*
tanh benchmark - N = 268435456 elements (1073.74 MB)
Average kernel time: 12.933 ms
Effective bandwidth: 249.07 GB/s
Correctness: PASSED
*/
/// V2
///V4
/*
tanh benchmark - N = 268435456 elements (1073.74 MB)
Average kernel time: 13.716 ms
Effective bandwidth: 234.85 GB/s
Correctness: PASSED
*/
///V4
/*

nvcc -O3 -arch=sm_86      -use_fast_math      -Xptxas -O3,-v      -lineinfo      -o tanh_harness tanh_harness.cu

*/

// ====================== YOUR KERNEL (expert version) ======================
__global__ void tanh_kernel_naive(const float* input, float* output, int N) {
    // Write code here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        output[idx] = (expf(input[idx]) - expf(-input[idx])) / (expf(input[idx]) + expf(-input[idx]));
    }
}

__global__ void tanh_kernel_v1(const float* input, float* output, int N) {
    // Write code here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        output[idx] = (__expf(input[idx]) - __expf(-input[idx])) / (__expf(input[idx]) + __expf(-input[idx]));
    }
}

__global__ void tanh_kernel_v2(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
   for (int i = idx; i < N; i += blockDim.x * gridDim.x){
        output[i] = (tanhf(input[i]) - tanhf(-input[i])) / (tanhf(input[i]) + tanhf(-input[i]));
    }
}

__global__ void tanh_kernel_v3(const float* input, float* output, int N) {
    // Write code here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x){
        output[i] = (__expf(input[i]) - __expf(-input[i])) / (__expf(input[i]) + __expf(-input[i]));
    }
}

__global__ void tanh_kernel_v4(const float* __restrict__ input,
                                  float* __restrict__ output, int N) {
    // Write code here
    int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    if (idx + 3 < N){
        float4 in = ((const float4*)input)[idx/4];
        float4 out;
        out.x = (tanhf(in.x));
        out.y = (tanhf(in.y));
        out.z = (tanhf(in.z));
        out.w = (tanhf(in.w));
        ((float4*)output)[idx/4] = out;
    }else {
        // Handling the remainder
        for (int i = 0; i < 4 && idx + i < N; ++i){
            int j = idx + i;
            output[j] = (tanhf(input[j]));
        } 
    }
}

extern "C" void solve_v1(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    tanh_kernel_v4<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}

__global__ void tanh_kernel_v5(const float* __restrict__ input,
                                  float* __restrict__ output, int N) {
    // Write code here
    int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    if (idx + 3 < N){
        float4 in = ((const float4*)input)[idx/4];
        float4 out;
#pragma unroll
        for (int i = 0; i < 4; ++i){
            float x = (&in.x)[i];
            (&out.x)[i] = tanhf(x);
        }
        ((float4*)output)[idx/4] = out;
    }else {
        // Handling the remainder
        for (int i = 0; i < 4 && idx + i < N; ++i){
            int j = idx + i;
            output[j] = (tanhf(input[j]));
        } 
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads*4 - 1) / (threads*4);
    tanh_kernel_v5<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}


// ====================== HARNESS ======================
int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 10000000;  // default 10M elements
    printf("tanh benchmark - N = %d elements (%.2f MB)\n", N, N*4.0/1e6);

    // Host pinned memory
    float *h_input, *h_output;
    cudaMallocHost(&h_input, N * sizeof(float));
    cudaMallocHost(&h_output, N * sizeof(float));

    // Initialize
    for (int i = 0; i < N; i++) {
        h_input[i] = rand() / (float)RAND_MAX;
    }

    // Device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Warm-up (5 launches)
    for (int i = 0; i < 10; i++) {
        solve(d_input, d_output, N);
    }

    // Timing with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int NUM_ITER = 1000;
    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITER; i++) {
        solve(d_input, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time_ms = milliseconds / NUM_ITER;
    float avg_time_s  = avg_time_ms / 1000.0f;

    // Effective bandwidth
    double bytes_moved = 3.0 * N * sizeof(float);           // read A, read B, write C
    double bw_gbps = (bytes_moved / avg_time_s) * 1e-9;

    printf("Average kernel time: %.3f ms\n", avg_time_ms);
    printf("Effective bandwidth: %.2f GB/s\n", bw_gbps);

    // Correctness check
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N && correct; i++) {
        float expected = tanh(h_input[i]);
        if (fabs(h_output[i] - expected) > 1e-5f) {
            correct = false;
            printf("Mismatch at index %d: %.6f vs %.6f\n", i, h_output[i], expected);
        }
    }
    printf("Correctness: %s\n", correct ? "PASSED" : "FAILED");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input); cudaFree(d_output);
    cudaFreeHost(h_input); cudaFreeHost(h_output);

    return 0;
}
