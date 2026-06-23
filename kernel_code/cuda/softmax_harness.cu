#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>

// nvcc -O3 -arch=sm_86      -use_fast_math      -Xptxas -O3,-v      -lineinfo      -o softmax_harness softmax_harness.cu
// 268435456
// ====================== YOUR KERNEL (expert version) ======================

__global__ void softmax_kernel_naive(const float* input, float* output, int N) {
    // Write code here
    int row = blockDim.x * blockIdx.x + threadIdx.x;float x_max = -INFINITY;
    
    float norm = 0.0f;
    
    for (int col = 0; col < N; col++){
        int i = row * N + col;
        x_max = max(x_max, input[i]);
    }
    for (int col = 0; col < N; col++){
        int i = row * N + col;
        norm += expf(input[i] - x_max);
    }
    for (int col = 0; col < N; col++){
        int i = row * N + col;
        output[i] = expf(input[i] - x_max) / norm;
    }
    // Max Redux
    __syncthreads();
    // Sum of Shifted Exponents}

}

__global__ void softmax_kernel(const float* input, float* output, int N) {
    // Write code here
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    // Find the Max
    float x_max = -INFINITY;
    if (idx == 0){
        for (int i = 0; i < N; ++i){
            x_max = fmax(x_max, input[i]);
        }
    }
    //Broadcast max to all threads
    __shared__ float s_max;
    if (idx == 0) s_max = x_max;
    __syncthreads();
    x_max = s_max;
    float sum = 0.0f;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x){
        sum += expf(input[i] - x_max);
    }
    //Reduce sum across all threads (simple atomic)
    atomicAdd(&output[0], sum);
    __syncthreads();

    if (idx == 0){
        float total_sum = output[0];
        output[0] = 0.0f;
        float logsum = x_max + logf(total_sum);

        for (int i = 0; i < N; ++i){
            output[i] = expf(input[i] - logsum);
        }
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}

static void softmax(float *input, size_t input_len) {
  //assert(input);
  // assert(input_len >= 0);  Not needed

  float m = -INFINITY;
  for (size_t i = 0; i < input_len; i++) {
    if (input[i] > m) {
      m = input[i];
    }
  }

  float sum = 0.0;
  for (size_t i = 0; i < input_len; i++) {
    sum += expf(input[i] - m);
  }

  float offset = m + logf(sum);
  for (size_t i = 0; i < input_len; i++) {
    input[i] = expf(input[i] - offset);
  }
}


// ====================== HARNESS ======================
int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 10000000;  // default 10M elements
    printf("softmax benchmark - N = %d elements (%.2f MB)\n", N, N*4.0/1e6);

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
    softmax(h_input, N);
    for (int i = 0; i < N && correct; i++) {
        float expected = h_input[i];
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
