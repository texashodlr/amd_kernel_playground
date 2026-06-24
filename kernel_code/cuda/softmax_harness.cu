#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>

// ====================== KERNEL ======================

__global__ void softmax_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int N)
{
    if (N <= 0) return;

    const int tid        = threadIdx.x;
    const int block_size = blockDim.x;
    const int stride     = block_size;

    // Pass 1: Max reduction
    float x_max = -INFINITY;
    for (int i = tid; i < N; i += stride) {
        x_max = fmaxf(x_max, input[i]);
    }

    __shared__ float s_max[1024];
    s_max[tid] = x_max;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    x_max = s_max[0];

    // Pass 2: Sum of exp(x - max)
    float sum = 0.0f;
    for (int i = tid; i < N; i += stride) {
        sum += expf(input[i] - x_max);
    }

    __shared__ float s_sum[1024];
    s_sum[tid] = sum;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    float total = s_sum[0];

    // Pass 3: Normalize
    float log_total = x_max + logf(total);
    for (int i = tid; i < N; i += stride) {
        output[i] = expf(input[i] - log_total);
    }
}

// ====================== SOLVE ======================
extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;

    // IMPORTANT: Only 1 block!
    softmax_kernel<<<1, 256>>>(input, output, N);
    cudaDeviceSynchronize();
}

// ====================== CPU REFERENCE ======================
static void softmax(float *input, size_t input_len) {
    float m = -INFINITY;

    // Phase 1: Finding a max given the array of inputs
    for (size_t i = 0; i < input_len; i++) {
        if (input[i] > m) m = input[i];
    }

    // Phase 2: Define the sum and add exponential (i-max)
    float sum = 0.0;
    for (size_t i = 0; i < input_len; i++) {
        sum += expf(input[i] - m);
    }

    // Phase 3: Exponentiating the input with the negative offset
    float offset = m + logf(sum);
    for (size_t i = 0; i < input_len; i++) {
        input[i] = expf(input[i] - offset);
    }
}

// ====================== HARNESS ======================
int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 10'000'000;

    printf("softmax benchmark - N = %d elements (%.2f MB)\n", N, N*4.0/1e6);

    float *h_input, *h_output;
    cudaMallocHost(&h_input,  N * sizeof(float));
    cudaMallocHost(&h_output, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_input[i] = rand() / (float)RAND_MAX * 10.0f - 5.0f;  // wider range
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input,  N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Warm-up
    for (int i = 0; i < 10; i++) {
        solve(d_input, d_output, N);
    }

    // Timing
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

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / NUM_ITER;

    double bytes = 3.0 * N * sizeof(float);
    double bw = (bytes / (avg_ms / 1000.0)) * 1e-9;

    printf("Average kernel time: %.3f ms\n", avg_ms);
    printf("Effective bandwidth: %.2f GB/s\n", bw);

    // Correctness
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    softmax(h_input, N);  // CPU reference

    bool correct = true;
    for (int i = 0; i < N && correct; i++) {
        if (fabs(h_output[i] - h_input[i]) > 1e-5f) {
            correct = false;
            printf("Mismatch at %d: %.8f vs %.8f\n", i, h_output[i], h_input[i]);
        }
    }
    printf("Correctness: %s\n", correct ? "PASSED" : "FAILED");

    // Cleanup
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_input); cudaFree(d_output);
    cudaFreeHost(h_input); cudaFreeHost(h_output);

    return 0;
}