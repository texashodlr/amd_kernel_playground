#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <cmath>
#include <iostream>

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

#define CEIL_DIV(M, N) ((M) + (N)-1) / (N)

void randomize_matrix(float *mat, int N) {
    struct timeval time;
    gettimeofday(&time, NULL);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++) {
        float tmp = (float) (rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}

void copy_matrix(float *src, float *dest, int N) {
    int i;
    for (i = 0; src + i && dest + i && i < N; i++)
        *(dest + i) = *(src + i);
    if (i != N)
        printf("copy failed at %d while there are %d elements in total.\n", i, N);
}

void print_matrix(const float *A, int M, int N) {
    int i;
    printf("[");
    for (i = 0; i < M * N; i++) {
        if ((i + 1) % N == 0)
            printf("%5.2f ", A[i]);
        else
            printf("%5.2f, ", A[i]);
        if ((i + 1) % N == 0) {
            if (i + 1 < M * N)
                printf(";\n");
        }
    }
    printf("]\n");
}

__global__ void gpu_mat_mul_v1(int M, int N, int K,
                               float alpha, float *A, float *B,
                               float beta, float *C)
{
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idy = blockIdx.y * blockDim.y + threadIdx.y;

    float tmp = 0.;

    if (idy >= M || idx >= N)
        return;

    for (int i = 0; i < K; i++) {
        tmp += A[idy * K + i] * B[i * N + idx];
    }
    C[idy * N + idx] = alpha * tmp + beta * C[idy * N + idx];
}

void cpu_mat_mul(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

bool verify_matrix(float *mat1, float *mat2, int N) {
    double diff = 0.0;
    int i;
    for (i = 0; mat1 + i && mat2 + i && i < N; i++) {
        diff = fabs((double) mat1[i] - (double) mat2[i]);
        if (diff > 1e-2) {
            printf("error. %5.2f,%5.2f,%d\n", mat1[i], mat2[i], i);
            return false;
        }
    }
    return true;
}

int main() {
    HIP_CHECK(hipInit(0));
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
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

    // Matrix Timing

    //Mmatrix size
    constexpr int size_len = 24;
    int SIZE[size_len];
    for (int i = 0; i < size_len; i++)
        SIZE[i] = 256 * (i + 1);

    int m, n, k, max_size;
    max_size = SIZE[size_len - 1];
    printf("max_size=%d\n", max_size);

    float alpha = 1.0, beta = 0.; //two arbitary input parameters，C=α*AB+β*C

    float *A = NULL, *B = NULL, *C = NULL, *C_ref = NULL;     //host matrices
    float *dA = NULL, *dB = NULL, *dC = NULL, *dC_ref = NULL; //device matrices

    A = (float *) malloc(sizeof(float) * max_size * max_size);
    B = (float *) malloc(sizeof(float) * max_size * max_size);
    C = (float *) malloc(sizeof(float) * max_size * max_size);
    C_ref = (float *) malloc(sizeof(float) * max_size * max_size);

    randomize_matrix(A, max_size * max_size);
    randomize_matrix(B, max_size * max_size);
    randomize_matrix(C, max_size * max_size);
    copy_matrix(C, C_ref, max_size * max_size);

    HIP_CHECK(hipMalloc((void **) &dA, sizeof(float) * max_size * max_size));
    HIP_CHECK(hipMalloc((void **) &dB, sizeof(float) * max_size * max_size));
    HIP_CHECK(hipMalloc((void **) &dC, sizeof(float) * max_size * max_size));
    HIP_CHECK(hipMalloc((void **) &dC_ref, sizeof(float) * max_size * max_size));

    hipStream_t stream;
    HIP_CHECK( hipStreamCreate(&stream));

    HIP_CHECK(hipMemcpyAsync(dA, A, sizeof(float) * max_size * max_size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(dB, B, sizeof(float) * max_size * max_size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(dC, C, sizeof(float) * max_size * max_size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(dC_ref, C_ref, sizeof(float) * max_size * max_size, hipMemcpyHostToDevice, stream));

    hipEvent_t start, stop;

    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));



    int repeat_times = 10;
    for (int i = 0; i < size_len; i++) {
        m = n = k = SIZE[i];
        dim3 block(32, 32, 1);
        dim3 grid(CEIL_DIV(n, block.x), CEIL_DIV(m, block.y), 1);
        int banana = 1;
        printf("m=n=k=%d\n", m);
        if ( banana != 0) {
            hipLaunchKernelGGL(gpu_mat_mul_v1, grid, block, 0, stream, m, n, k, alpha, dA, dB, beta, dC); // user define
            cpu_mat_mul(m, n, k, alpha, A, B, beta, C_ref);
            HIP_CHECK(hipMemcpyAsync(C, dC, sizeof(float) * m * n, hipMemcpyDeviceToHost, stream));

            if (!verify_matrix(C_ref, C, m * n)) {
                printf("Failed to pass the correctness verification against NVIDIA cuBLAS. Exited.\n");
                exit(EXIT_FAILURE);
            }
        }
        HIP_CHECK( hipStreamSynchronize(stream));

        HIP_CHECK(hipEventRecord(start, stream));
        for (int j = 0; j < repeat_times; j++) {
            hipLaunchKernelGGL(gpu_mat_mul_v1, grid, block, 0, stream, m, n, k, alpha, dA, dB, beta, dC); // user define
        }
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));
        float elapsedMs = 0.0f;
        HIP_CHECK( hipEventElapsedTime(&elapsedMs, start, stop) );
        float elapsedSec = elapsedMs / 1000.;
        double avgSec = elapsedSec / repeat_times;

        double gflops = (2.0 * repeat_times * m * n * k) / elapsedSec / 1.0e9;

        printf("Average elasped time: (%f) second, performance: (%f) GFLOPS. size: (%d).\n", avgSec, gflops, m);
        fflush(stdout);
        copy_matrix(C_ref, C, m * n); //sync C with cuBLAS to prepare for the next run
    }
        /* ===== Cleanup ===== */
    HIP_CHECK( hipEventDestroy(start) );
    HIP_CHECK( hipEventDestroy(stop) );
    HIP_CHECK( hipStreamDestroy(stream) );

    HIP_CHECK( hipFree(dA) );
    HIP_CHECK( hipFree(dB) );
    HIP_CHECK( hipFree(dC) );
    HIP_CHECK( hipFree(dC_ref) );
    free(A);
    free(B);
    free(C);
    free(C_ref);

    return 0;
};