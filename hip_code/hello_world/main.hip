#include "example_utils.hip"
#include <hip/hip_runtime.h>
#include <iostream>

__device__ unsigned int get_thread_idx(){
	return threadIdx.x;
}

__host__ void print_hello_host(){
	std::cout << "Hello world from host!" << std::endl;
}

__device__ __host__ void print_hello(){
	printf("Hello world from device or host!\n");
}

__global__ void helloworld_kernel(){
	unsigned int thread_idx = get_thread_idx();
	unsigned int block_idx  = blockIdx.x;
	print_hello();
	printf("Hello world from device kernel block %u thread %u!\n", block_idx, thread_idx);
}

int main(){
	print_hello_host();
	print_hello();
	helloworld_kernel<<<dim3(2),
			    dim3(2),
			    0,
			    hipStreamDefault
			    >>>();
	//HIP_CHECK(hipDeviceSynchronize());
}
