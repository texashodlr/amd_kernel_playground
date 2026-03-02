#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <algorithm>


#define HIP_CHECK(call)                                                     \
  do {                                                                      \
    hipError_t _e = (call);                                                 \
    if (_e != hipSuccess) {                                                 \
      std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,     \
                   hipGetErrorString(_e));                                  \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

static inline double gib(double bytes) {return bytes / (1024.0 * 1024.0 * 1024.0); }

static void print_device_props(int dev = 0){
  
  // Device Query Section BEGIN
  // Source: https://rocm.docs.amd.com/projects/HIP/en/docs-5.6.1/.doxygen/docBin/html/structhip_device_prop__t.html
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));

  printf("                     Device: %s\n", prop.name);
  printf("                gcnArchName: %s\n", prop.gcnArchName);
  printf("  multiProcessorCount (CUs): %d\n", prop.multiProcessorCount);
  printf("       warpSize (wavefront): %d\n", prop.warpSize);
  printf("               regsPerBlock: %d\n", prop.regsPerBlock);
  printf("         maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
  printf("maxThreadsPerMultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
  printf("             totalGlobalMem: %.2f GiB\n",gib((double)prop.totalGlobalMem));
  printf("                  clockRate: %d kHz\n", prop.clockRate);

  // Device Query Section END
}

__global__ void hello_kernel(int* out) {
  // Only one thread prints to avoid spam
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("Hello from HIP kernel! grid=(%d) block=(%d)\n",
           gridDim.x, blockDim.x);
  }
  // Write something so we can validate we ran
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  out[idx] = idx;
}

// HIP Timing Harness
static float time_ms(const std::function<void()>& fn, int iters = 200) {
  hipEvent_t start{}, stop{};
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipEventRecord(start, nullptr));
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  HIP_CHECK(hipEventRecord(stop, nullptr));
  HIP_CHECK(hipEventSynchronize(stop));

  float ms_total = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms_total, start, stop));

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  return ms_total / iters;
}

// Kernel #1: Pure bandwidth kernel
__global__ void copy_kernel(const float* __restrict__ a,
                            float* __restrict__ b,
                            int n){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = tid; i < n; i += stride) b[i] = a[i];
}

// Kernel #2: Compute Heavy Kernel
__global__ void fma_kernel(float* out, int n, int iters){
  int i = blockIdx.x * blockDim.x * threadIdx.x;
  if (i < n){
    float x = (float)i * 0.000001f + 1.0f;
    float a = 1.000001f;
    float b = 0.000001f;
  #pragma unroll 4
    for (int k = 0; k < iters; ++k){
      x = x * a + b;
    }
    out[i] = x;
  }
}

// Init Kernel
__global__ void init_kernel(float* buf, int n, float v){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < n; i += stride) buf[i] = v;
}

static void warmup(){
  HIP_CHECK(hipDeviceSynchronize());
}

static void ensure_reasonable_device_buffers(size_t bytes_needed){
  size_t free_b = 0, total_b = 0;
  HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
  if (bytes_needed > free_b * 0.90){
    std::fprintf(stderr,
                 "Requested %.2f GiB but only %.2f GiB free on device. "
                 "Reduce N.\n",
                 gib((double)bytes_needed), gib((double)free_b));
    std::exit(3);
  }
}


static void bench_copy(int CUs){
  std::printf("== Benchmark A: Streaming copy (GB/s) ==\n");
  const int N = 256 * 1024 * 1024;
  const size_t bytes = (size_t)N * sizeof(float);

  ensure_reasonable_device_buffers(2 * bytes);

  float* d_a = nullptr;
  float* d_b = nullptr;
  HIP_CHECK(hipMalloc(&d_a, bytes));
  HIP_CHECK(hipMalloc(&d_b, bytes));

  //Initialization to allocate
  {
    int tpb = 256;
    int blocks = std::max(1, CUs * 8);
    hipLaunchKernelGGL(init_kernel, dim3(blocks), dim3(tpb), 0, 0, d_a, N, 1.0f);
    hipLaunchKernelGGL(init_kernel, dim3(blocks), dim3(tpb), 0, 0, d_b, N, 0.0f);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  const std::vector<int> block_sizes = {64, 128, 256, 512, 1024};
  const std::vector<int> blocks_per_cu = {1, 2, 4, 8, 16};

  std::printf("N = %d floats (%.2f GiB per buffer)\n", N, gib((double)bytes));
  std::printf("%10s %14s %14s\n", "TPB", "Blocks", "GB/s");

  for (int tpb : block_sizes) {
    for (int bpcu : blocks_per_cu) {
      int blocks  = std::max(1, CUs * bpcu);
      auto fn = [&](){
        hipLaunchKernelGGL(copy_kernel, dim3(blocks), dim3(tpb), 0, 0, d_a, d_b, N);
      };
      float ms = time_ms(fn, /*iters=*/30);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());

      // Minimum traffic assumption: 8 bytes per element (read+write float).
      double bytes_moved = (double)N * 8.0;
      double gbps = (bytes_moved / (ms / 1000.0)) / 1e9;

      std::printf("%10d %14d %14.2f\n", tpb, blocks, gbps);
    }
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  std::printf("\n");
}

static void bench_fma(int CUs){
  std::printf("== Benchmark B: FMA loop (approx GFLOP/s) ==\n");
  const int N = 64 * 1024 * 1024;
  const size_t bytes = (size_t)N * sizeof(float);

  ensure_reasonable_device_buffers(bytes);
  float* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, bytes));
  // Init
  {
    int tpb = 256;
    int blocks = std::max(1, CUs * 8);
    hipLaunchKernelGGL(init_kernel, dim3(blocks), dim3(tpb), 0, 0, d_out, N, 0.0f);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  const std::vector<int> block_sizes = {64, 128, 256, 512, 1024};
  const std::vector<int> blocks_per_cu = {1, 2, 4, 8, 16};
  const std::vector<int> iters_list = {64, 256, 1024, 4096};

  std::printf("N = %d floats (%.2f GiB output)\n", N, gib((double)bytes));
  std::printf("%10s %14s %10s %14s\n", "TPB", "Blocks", "Iters", "GFLOP/s");

  for (int iters: iters_list) {
    for (int tpb: block_sizes) {
      for (int bpcu: blocks_per_cu){
        int blocks = std::max(1, CUs * bpcu);
        auto fn = [&](){
        hipLaunchKernelGGL(fma_kernel, dim3(blocks), dim3(tpb), 0, 0, d_out, N, iters);
      };

      float ms = time_ms(fn, 80);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());

      double flops = (double)N * (double)iters * 2.0;
      double gflops = (flops / (ms / 1000.0)) / 1e9;

      std::printf("%10d %14d %10d %14.2f\n", tpb, blocks, iters, gflops);
      }
    }
    std::printf("\n");
  }
  HIP_CHECK(hipFree(d_out));
  std::printf("\n");
}

int main() {
  int dev = 0;
  HIP_CHECK(hipSetDevice(dev));

  print_device_props(dev);

  hipDeviceProp_t prop{};
  
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  int CUs = prop.multiProcessorCount;

  // Simple warmup to avoid first-run penalities
  warmup();

  bench_copy(CUs);
  bench_fma(CUs);

  std::printf("Done.\n");
  return 0;
}