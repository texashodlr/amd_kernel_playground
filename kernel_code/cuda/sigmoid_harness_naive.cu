#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
// nvcc -O3 -arch=sm_86 -o sigmoid_harness_naive sigmoid_harness_naive.cu
// 268435456

/// NAIVE
/*
Sigmoid benchmark - N = 268435456 elements (1073.74 MB)
Average kernel time: 16.284 ms
Effective bandwidth: 197.81 GB/s
Correctness: PASSED
*/
/// NAIVE
/// V1
/*
Sigmoid benchmark - N = 268435456 elements (1073.74 MB)
Average kernel time: 13.087 ms
Effective bandwidth: 246.14 GB/s
Correctness: PASSED
*/
/// V1
/// V1 with compile flags: -ftz=true -prec-div=false -prec-sqrt=false
/*
Sigmoid benchmark - N = 268435456 elements (1073.74 MB)
Average kernel time: 12.847 ms
Effective bandwidth: 250.75 GB/s
Correctness: PASSED
*/
/// V1 with compile flags
/// v2 using tanhf
/*
Sigmoid benchmark - N = 268435456 elements (1073.74 MB)
Average kernel time: 12.947 ms
Effective bandwidth: 248.81 GB/s
Correctness: PASSED
*/
///
/// v3 with grid-stride
/*
Sigmoid benchmark - N = 268435456 elements (1073.74 MB)
Average kernel time: 12.785 ms
Effective bandwidth: 251.95 GB/s
Correctness: PASSED
*/
///v3 with grid-stride
/// V4 Vectorized Memory Acces
/*
Sigmoid benchmark - N = 268435456 elements (1073.74 MB)
Average kernel time: 15.705 ms
Effective bandwidth: 205.11 GB/s
Correctness: PASSED
*/
/// V4 Vectorized Memory Acces
__forceinline__ __device__ float __tanhf(float x) {
    float r;
    // Uses the MUFU.TANH instruction on Compute Capability 7.5+
    asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}   

// ====================== YOUR KERNEL (expert version) ======================
__global__ void sigmoid_kernel_naive(const float* input, float* output, int N) {
    // Write code here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        output[idx] = (1.0 / (1.0 + expf(-input[idx])));
    }
}

__global__ void sigmoid_kernel_v1(const float* input, float* output, int N) {
    // Write code here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        output[idx] = (1.0f / (1.0f + __expf(-input[idx])));
    }
}

__global__ void sigmoid_kernel_v2(const float* input, float* output, int N) {
    // Write code here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        float x = input[idx];
        output[idx] = 0.5f * (1.0f + __tanhf(0.5f * x));
    }
}

__global__ void sigmoid_kernel_v3(const float* input, float* output, int N) {
    // Write code here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x){
        output[idx] = (1.0f / (1.0f + __expf(-input[idx])));
    }
}

__global__ void sigmoid_kernel_v4(const float* __restrict__ input,
                                  float* __restrict__ output, int N) {
    // Write code here
    int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    if (idx + 3 < N){
        float4 in = ((const float4*)input)[idx/4];
        float4 out;
        out.x = 1.0 / (1.0f + __expf(-in.x));
        out.y = 1.0 / (1.0f + __expf(-in.y));
        out.z = 1.0 / (1.0f + __expf(-in.z));
        out.w = 1.0 / (1.0f + __expf(-in.w));
        ((float4*)output)[idx/4] = out;
    }else {
        // Handling the remainder
        for (int i = 0; i < 4 && idx + i < N; ++i){
            int j = idx + i;
            output[j] = (1.0f / (1.0f + __expf(-input[j])));
        } 
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    sigmoid_kernel_v3<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}

extern "C" void solve_v4(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads*4 - 1) / threads*4;
    sigmoid_kernel_v3<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}


// ====================== HARNESS ======================
int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 10000000;  // default 10M elements
    printf("Sigmoid benchmark - N = %d elements (%.2f MB)\n", N, N*4.0/1e6);

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
        solve_v4(d_input, d_output, N);
    }

    // Timing with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int NUM_ITER = 1000;
    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITER; i++) {
        solve_v4(d_input, d_output, N);
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
        float expected = (1.0 / (1.0 + exp(-h_input[i])));
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
