#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ====================== YOUR KERNEL (expert version) ======================

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    // Write code here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        C[idx] = A[idx] + B[idx]; 
    }
}
// ====================== YOUR WRAPPER ======================
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threads = 256;                    // excellent starting point on Ampere
    int blocks = (N + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();              // ensures timing is clean
}

// ====================== HARNESS ======================
int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 10000000;  // default 10M elements
    printf("VectorAdd benchmark - N = %d elements (%.2f MB)\n", N, N*4.0/1e6);

    // Host pinned memory
    float *h_A, *h_B, *h_C;
    cudaMallocHost(&h_A, N * sizeof(float));
    cudaMallocHost(&h_B, N * sizeof(float));
    cudaMallocHost(&h_C, N * sizeof(float));

    // Initialize
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Warm-up (5 launches)
    for (int i = 0; i < 5; i++) {
        solve(d_A, d_B, d_C, N);
    }

    // Timing with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int NUM_ITER = 100;
    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITER; i++) {
        solve(d_A, d_B, d_C, N);
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
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N && correct; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            correct = false;
            printf("Mismatch at index %d: %.6f vs %.6f\n", i, h_C[i], expected);
        }
    }
    printf("Correctness: %s\n", correct ? "PASSED" : "FAILED");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);

    return 0;
}
