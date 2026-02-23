#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(call)                                                     \
  do {                                                                      \
    hipError_t _e = (call);                                                 \
    if (_e != hipSuccess) {                                                 \
      std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,     \
                   hipGetErrorString(_e));                                  \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

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

int main() {
  // Pick a small launch
  constexpr int blocks = 2;
  constexpr int threads = 128;
  constexpr int N = blocks * threads;

  int* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, N * sizeof(int)));

  // Launch kernel
  hipLaunchKernelGGL(hello_kernel,
                     dim3(blocks),
                     dim3(threads),
                     0,   // shared mem
                     0,   // stream
                     d_out);

  // Make sure kernel completed and device-side printf flushed
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // Copy results back and sanity-check
  int* h_out = (int*)std::malloc(N * sizeof(int));
  HIP_CHECK(hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < N; ++i) {
    if (h_out[i] != i) {
      std::fprintf(stderr, "Mismatch at %d: got %d expected %d\n", i, h_out[i], i);
      ok = false;
      break;
    }
  }

  std::printf("Validation: %s\n", ok ? "PASS" : "FAIL");

  std::free(h_out);
  HIP_CHECK(hipFree(d_out));
  return ok ? 0 : 2;
}