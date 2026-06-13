#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

// ====================== YOUR KERNEL (expert version) ======================

__global__ void naive_relu_kernel(const float* input, float* output, int N) {
    // Write code here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        if (input[idx] <= 0.0){
            output[idx] = 0.0;
        }
        else{
            output[idx] = input[idx];
        }
    }
}

__global__ void relu_kernel(const float* input, float* output, int N) {
    // Write code here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        output[idx] = fmaxf(0.0, input[idx]);
    }
}

// ====================== YOUR WRAPPER ======================
extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}

// ====================== HARNESS ======================
int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 10000000;  // default 10M elements
    printf("VectorAdd benchmark - N = %d elements (%.2f MB)\n", N, N*4.0/1e6);

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

    cudaMemcpy(d_input, d_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Warm-up (5 launches)
    for (int i = 0; i < 5; i++) {
        solve(d_input, d_output, N);
    }

    // Timing with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int NUM_ITER = 100;
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
        float expected = max(0.0, h_input[i]);
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
