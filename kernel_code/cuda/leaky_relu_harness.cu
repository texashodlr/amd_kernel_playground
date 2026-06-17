#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>

// nvcc -O3 -arch=sm_86 -o sigmoid_harness_naive sigmoid_harness_naive.cu
// 268435456

/*
nvcc -O3 -arch=sm_86      -use_fast_math      -Xptxas -O3,-v      -lineinfo      -o leaky_relu_harness.cu
*/

//Naive
/*
leaky relu benchmark - N = 268435456 elements (1073.74 MB)
Average kernel time: 12.969 ms
Effective bandwidth: 248.37 GB/s
Correctness: PASSED
*/
//V2
/*
leaky relu benchmark - N = 268435456 elements (1073.74 MB)
Average kernel time: 12.942 ms
Effective bandwidth: 248.89 GB/s
Correctness: PASSED
*/
//V3
/*
leaky relu benchmark - N = 268435456 elements (1073.74 MB)
Average kernel time: 12.924 ms
Effective bandwidth: 249.24 GB/s
Correctness: PASSED
*/
//V4
/*
leaky relu benchmark - N = 268435456 elements (1073.74 MB)
Average kernel time: 12.948 ms
Effective bandwidth: 248.78 GB/s
Correctness: PASSED
*/
//V4
/*
leaky relu benchmark - N = 268435456 elements (1073.74 MB)
Average kernel time: 14.694 ms
Effective bandwidth: 219.22 GB/s
Correctness: PASSED
*/

// ====================== YOUR KERNEL (expert version) ======================
__global__ void leaky_relu_kernel_v1(const float* input, float* output, float alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        output[idx] = (input[idx] >= 0) ? input[idx] : (alpha * input[idx]);
    }
}

__global__ void leaky_relu_kernel_v2(const float* input, float* output, float alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        output[idx] = fmaxf(input[idx], alpha* input[idx]);
    }
}

__global__ void leaky_relu_kernel_v3(const float* __restrict__ input, float* __restrict__ output, float alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (int i = idx; i < N; i += blockDim.x * gridDim.x){
        output[i] = fmaxf(input[i], alpha* input[i]);
    }
}

__global__ void leaky_relu_kernel_v4(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     float alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x * 4;  // 4 elements per iteration

    // Main vectorized loop
    for (int i = idx * 4; i + 3 < N; i += stride) { 
        float4 in = ((const float4*)input)[i / 4];
        float4 out;
        out.x = fmaxf(in.x, alpha * in.x);
        out.y = fmaxf(in.y, alpha * in.y);
        out.z = fmaxf(in.z, alpha * in.z);
        out.w = fmaxf(in.w, alpha * in.w);
        ((float4*)output)[i / 4] = out;
    }
    for (int j = 0; j < 4; ++j) { 
        int i = idx * 4 + j;
        if (i < N) {
            output[i] = fmaxf(input[i], alpha * input[i]);
        }
    }
}


extern "C" void solve(const float* input, float* output, float alpha, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads; 
    leaky_relu_kernel_v4<<<blocks, threads>>>(input, output, alpha, N);
    cudaDeviceSynchronize(); 
}


// ====================== HARNESS ======================
int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 10000000;  // default 10M elements
    int alpha = (argc > 1) ? atoi(argv[2]) : 0.01;
    printf("leaky relu benchmark - N = %d elements (%.2f MB)\n", N, N*4.0/1e6);

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
        solve(d_input, d_output, alpha, N);
    }

    // Timing with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int NUM_ITER = 1000;
    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITER; i++) {
        solve(d_input, d_output, alpha, N);
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
        float expected = (h_input[i] >= 0)? h_input[i] : (alpha * h_input[i]);
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
