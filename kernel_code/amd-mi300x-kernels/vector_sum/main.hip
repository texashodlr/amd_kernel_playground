#include "example_utils.hpp"
#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <cmath>

#include <cstddef>

__global__ void vector_sum_kernel_naive(const float* a, const float* b, float* c, const unsigned int size){
    const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx < size)
    {
        c[global_idx] = a[global_idx] + b[global_idx];
    }
    __syncthreads();
    if ((global_idx + 1) == size){
        // Naive for-loop that uses final thread to recompute everything baby
        for (unsigned int i = 1; i < size; i++){
            c[0] += c[i];
        }      
    }
}

bool validate_vector_sum_kernel(const float* host_a, const float* host_b, const float* gpu_c, unsigned int size){
    constexpr float epsilon = 1e-5f;
    float cpu_res = 0.0;
    for (unsigned int i = 0; i < size; ++i){
        cpu_res += host_a[i] + host_b[i];
    }
    if (std::fabs(cpu_res - gpu_c[0]) > epsilon) {
        std::cerr << "\n[FAIL] Mismatch found at index " << "0" << "!\n"
                    << "  CPU Calc:  " << cpu_res << "\n"
                    << "  GPU Calc:  " << gpu_c[0] << "\n"
                    << "  Difference: " << std::fabs(cpu_res - gpu_c[0]) << "\n";
        return false;
    }
    return true;
}

int main(){

    /*
    Implement a vector sum reduction kernel. 
    This kernel computes the sum of all elements in the input tensor.
    Input: A tensor of shape (N,) with values from a normal distribution
        with mean 0 and variance 1. 
    Output: A scalar value equal to the sum of all elements in the input tensor.
    {"size":1638400}
    {"size":3276800}
    {"size":6553600}
    {"size":13107200}
    {"size":26214400}
    {"size":52428800}
    */

    constexpr unsigned int sizes[6] = {1638400, 3276800, 6553600, 13107200, 26214400, 52428800};

    constexpr size_t max_size = sizes[5] * sizeof(float);

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    HIP_CHECK(hipMalloc(&d_A, max_size));
    HIP_CHECK(hipMalloc(&d_B, max_size));
    HIP_CHECK(hipMalloc(&d_C, max_size));

    std::vector<float> h_A(max_size);
    std::vector<float> h_B(max_size);
    std::vector<float> h_C(max_size);

    // Psuedo random generator
    rocrand_generator prng;
    rocrand_create_generator(&prng, ROCRAND_RNG_PSEUDO_DEFAULT);
    rocrand_set_seed(prng, 42ULL);

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    std::cout << std::left << std::setw(15) << "Elements" 
              << std::setw(15) << "Memory (MB)" 
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "TpB (#)"
              << std::setw(15) << "BpG (#)" << "\n";
    std::cout << "------------------------------------------------------------------------------------------\n";

    // Kernel harness loop
    for (int i = 0; i < 6; ++i){
        unsigned int currentSize  = sizes[i];
        size_t bytesAllocated = currentSize * sizeof(float);
        double sizeInMB = static_cast<double>(bytesAllocated) / (1024.0*1024.0);

        unsigned int rngSize = (currentSize % 2 == 0) ? currentSize: currentSize + 1;

        rocrand_generate_normal(prng, d_A, rngSize, 0.0f, 1.0f);
        rocrand_generate_normal(prng, d_B, rngSize, 0.0f, 1.0f);

        int threadsPerBlock = 256;
        int blocksPerGrid = (currentSize + threadsPerBlock - 1) / threadsPerBlock;

        HIP_CHECK(hipEventRecord(start, nullptr));

        vector_sum_kernel_naive<<<dim3(blocksPerGrid), dim3(threadsPerBlock), 0, hipStreamDefault>>>(d_A, d_B, d_C, currentSize);

        HIP_CHECK(hipEventRecord(stop, nullptr));
        HIP_CHECK(hipEventSynchronize(stop));
        float event_ms;
        HIP_CHECK(hipEventElapsedTime(&event_ms, start, stop));

        // Validation of GPU kernel
        HIP_CHECK(hipMemcpy(h_A.data(), d_A, currentSize*sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_B.data(), d_B, currentSize*sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, currentSize*sizeof(float), hipMemcpyDeviceToHost));

        bool pass = validate_vector_sum_kernel(h_A.data(), h_B.data(), h_C.data(), currentSize);

        std::cout << std::left << std::setw(15) << currentSize 
                  << std::setw(15) << std::fixed << std::setprecision(2) << sizeInMB * 3 // Total data for 3 arrays
                  << std::setw(15) << std::setprecision(4) << event_ms
                  << std::setw(15) << threadsPerBlock
                  << std::setw(15) << blocksPerGrid << "\n";
    
        if (!pass) break;
    } 

    // Cleanup
    rocrand_destroy_generator(prng);
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    std::cout << "\nHarness execution comlete. Resources freed safely. \n";
    return 0;

}