#include </opt/rocm/core-7.14/hip_runtime.h>
#include <iostream>
#include <cstdlib>

#define HIP_CHECK(cmd) \
do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error: " \
                  << hipGetErrorString(err) \
                  << " at " << __FILE__ \
                  << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)


__global__ void vectorAdd(const float* A,
                          const float* B,
                          float* C,
                          int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    HIP_CHECK(hipInit(0));
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(%deviceCount));
    if(deviceCount == 0){
        std::cerr << "No HIP Devices available\n";
        return EXIT_FAILURE;
    }

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));

    std::cout << "GPU: " << props.name << "\n";
    std::cout << "Architecture: " << props.gcnArchName << "\n";
    std::cout << "Compute Units: " << props.multiProcessorCount << "\n";
    std::cout << "Global Memory: "
              << props.totalGlobalMem / (1024 * 1024) << "\n\n";
    
    const int N = 1024;
    size_t size = N * sizeof(float);

    float* hA = new float[N];
    float* hB = new float[N];
    float* hC = new float[N];

    for (int i = 0; i < N; i++){
        hA[i] = static_cast<float>(i);
        hB[i] = 2.0f * static_cast<float>(i);
    }

    float *dA, *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, size));
    HIP_CHECK(hipMalloc(&dB, size));
    HIP_CHECK(hipMalloc(&dC, size));
    HIP_CHECK(hipMemSet(dC, 0 , size));

    hipStream_t stream;
    HIP_CHECK( hipStreamCreate(&stream));

    HIP_CHECK( hipMemcpyAsync(dA,hA,size,hipMemcpyHostToDevice,stream));
    HIP_CHECK( hipMemcpyAsync(dB,hB,size,hipMemcpyHostToDevice,stream));

    hipEvent_t start, stop;

    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start,stream));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    hipLaunchKernelGGL(vectorAdd, grid, block, 0, stream, dA, dB, dC, N);
    HIP_CHECK( hipEventRecord(stop, stream) );
    HIP_CHECK( hipEventSynchronize(stop) );

    float elapsedMs = 0.0f;
    HIP_CHECK( hipEventElapsedTime(&elapsedMs, start, stop) );
    std::cout << "Kernel time (ms): " << elapsedMs << "\n";

    /* ===== Result Retrieval and Validation ===== */
    HIP_CHECK( hipMemcpyAsync(hC, dC, size,
                              hipMemcpyDeviceToHost, stream) );
    HIP_CHECK( hipStreamSynchronize(stream) );

    bool pass = true;
    for (int i = 0; i < N; i++) {
        if (hC[i] != hA[i] + hB[i]) {
            pass = false;
            break;
        }
    }

    std::cout << "Vector Add Result: "
              << (pass ? "PASS" : "FAIL") << "\n";

    /* ===== Cleanup ===== */
    HIP_CHECK( hipEventDestroy(start) );
    HIP_CHECK( hipEventDestroy(stop) );
    HIP_CHECK( hipStreamDestroy(stream) );

    HIP_CHECK( hipFree(dA) );
    HIP_CHECK( hipFree(dB) );
    HIP_CHECK( hipFree(dC) );

    delete[] hA;
    delete[] hB;
    delete[] hC;

    HIP_CHECK( hipDeviceReset() );

    return EXIT_SUCCESS;
}


/*

GPU: AMD Radeon(TM) 8060S Graphics
Architecture: gfx1151
Compute Units: 20
Global Memory (MB): 111049

Kernel time (ms): 2.89536
Vector Add Result: PASS
*/