#include "example_utils.hpp"
#include <hip/hip_runtime.h>

#include <iomanip>
#include <iostream>
#include <vector>


__global__ void pairwise_product_kernel(float* C, const float* A, const float* B, const unsigned int size){
    const unsigned int global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_thread_idx < size){
        C[global_thread_idx] = A[global_thread_idx] * B[global_thread_idx];
    }
}


void print_occupancy(const int block_size){
    hipDeviceProp_t dev_prop;
    HIP_CHECK(hipGetDeviceProperties(&dev_prop, 0));
    if(dev_prop.maxThreadsPerMultiProcessor){
        int num_blocks = 0;
        HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, pairwise_product_kernel, block_size, 0));
        std::cout << "Theoretical Occupancy is "
                  << double_precision(static_cast<double>(num_blocks) * block_size / dev_prop.maxThreadsPerMultiProcessor * 100, 3) << "%" << std::endl;
                 
    }
}

void deploy_kernel_manual_parameters(float* d_C,
                                    const float* d_A,
                                    const float* d_B,
                                    const unsigned int size){
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    constexpr unsigned int block_size = 32;

    const unsigned int grid_size = (size + (block_size - 1)) / block_size;
    std::cout << "\nManual Configuration with Block Size " << block_size << '\n';

    HIP_CHECK(hipEventRecord(start, nullptr));

    pairwise_product_kernel<<<dim3(grid_size), dim3(block_size), 0, hipStreamDefault>>>(d_C,d_A,d_B, size);
    HIP_CHECK(hipEventRecord(stop, nullptr));
    HIP_CHECK(hipEventSynchronize(stop));

    float event_ms;
    HIP_CHECK(hipEventElapsedTime(&event_ms, start, stop));
    std::cout << "Kernel Execution Time: " << double_precision(event_ms, 2) << "ms " << std::endl;

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    print_occupancy(block_size);
}

void deploy_kernel_automatic_parameters(float* d_C,
                                    const float* d_A,
                                    const float* d_B,
                                    const unsigned int size){
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    int min_grid_size = 0;
    int block_size = 0;

    HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, pairwise_product_kernel, 0,0));

    std::cout << "\nAutomatic Configuration based on hipOccupancyMaxPotentialBlockSize \n";
    std::cout << "Suggested Block Size is " << block_size << ", Minimum Grid Size is "<< min_grid_size << std::endl;

    const unsigned int grid_size = (size + (block_size - 1)) / block_size;

    HIP_CHECK(hipEventRecord(start, nullptr));

    pairwise_product_kernel<<<dim3(grid_size), dim3(block_size), 0, hipStreamDefault>>>(d_C,d_A,d_B, size);
    HIP_CHECK(hipEventRecord(stop, nullptr));
    HIP_CHECK(hipEventSynchronize(stop));

    float event_ms;
    HIP_CHECK(hipEventElapsedTime(&event_ms, start, stop));
    std::cout << "Kernel Execution Time: " << double_precision(event_ms, 2) << "ms " << std::endl;

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    print_occupancy(block_size);
}                                    

int main(){
    constexpr int size = 10000;
    
    std::vector<float> h_A(size);
    std::vector<float> h_B(size);

    for (int i = 0; i < size; i++){
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i);
    }

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    const size_t size_in_bytes = sizeof(float) * h_A.size();
    HIP_CHECK(hipMalloc(&d_A, size_in_bytes));
    HIP_CHECK(hipMalloc(&d_B, size_in_bytes));
    HIP_CHECK(hipMalloc(&d_C, size_in_bytes));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_in_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_in_bytes, hipMemcpyHostToDevice));

    deploy_kernel_manual_parameters(d_C, d_A, d_B, size);

    deploy_kernel_automatic_parameters(d_C, d_A, d_B, size);


    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
}