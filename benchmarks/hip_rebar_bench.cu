// HIP benchmark: Host <-> Device memory bandwidth with ReBAR validation
// Compile: hipcc -o host_device_bw host_device_bw.cu -lpthread
// Run: ./host_device_bw [iterations] [device_id]

#define _GNU_SOURCE
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <cmath>
#include <sched.h>
#include <numa.h>



#define CHECK_HIP(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

static double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int check_numa(){
    
    if (numa_available() < 0){
        printf("NUMA isn't available.\n");
        return -1;
    }
    int cpu, node;
    if (getcpu(&cpu, &node) == 0){
        printf("Current CPU: %d, Current NUMA Node: %d\n", cpu, node);
    }
    int max_numa = numa_max_node();
    printf("Max NUMA node: %d\n",max_node);

    return 0;
}

// Check ReBAR status
static void check_rebar(int device_id) {

    if (check_numa() < 0){
        printf("NUMA not detected.\n");
    }

    int rebar_support = 0;
    int rebar_enabled = 0;

    CHECK_HIP(hipDeviceGetAttribute(&rebar_support,
        hipDeviceAttributeIsLargeBar, device_id));

    // Check if memory space is actually available (indicates enabled)
    size_t free_mem, total_mem;
    CHECK_HIP(hipMemGetInfo(&free_mem, &total_mem));

    // On MI300X, if ReBAR is disabled, we typically see limited BAR aperture
    // The attribute tells us capability, not enabled state
    printf("=== ReBAR Status ===\n");
    printf("Device: %d\n", device_id);
    printf("ReBAR Support: %s\n", rebar_support ? "Yes" : "No");

    // Additional check: try mapping a large allocation to see if it works
    // Without ReBAR, large peer allocations may fail or use slow path
    void *test_ptr;
    for(int i = 2; i < 18; i++){
        int byte_size = pow(2,i);
        hipError_t err = hipMalloc(&test_ptr, byte_size * 1024 * 1024); // 512MB
        if (err == hipSuccess) {
            printf("Large allocation (%dMB): Success\n", byte_size);
            CHECK_HIP(hipFree(test_ptr));
        } else {
            printf("Large allocation (%dMB): Failed - %s\n",byte_size , hipGetErrorString(err));
        }
    }
    
    // HIP Timing harness
    float ms;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // NUMA Node Affinity transfer check
    int max_node = numa_max_node();
    for(int numa_node = 0; numa_node <= max_node; numa_node++){
        printf("Testing NUMA node %d...\n", numa_node);
        numa_run_on_node(numa_node); // Pin thread to NUMA node
        
        for(int i = 2; i < 11; i++){
            
            int byte_size = pow(2,i);
            size_t host_byte_size = byte_size * 1024 * 1024;

            void *host_buffer = numa_alloc_onnode(host_byte_size, numa_node); // Host memory on the node
            if (!host_buffer) {
                printf(" %dMB: numa_alloc_onnode failed\n", byte_size);
                continue;
            }
            // Register as pinned memory for DMA
            CHECK_HIP(hipHostRegister(host_buffer, host_byte_size, hipHostRegisterDefault));

            // Allocate GPU Memory
            void *device_buffer;
            CHECK_HIP(hipMalloc(&device_buffer, host_byte_size));

            // Transfer H2D
            hipEventRecord(start,0);
            CHECK_HIP(hipMemcpy(device_buffer, host_buffer, host_byte_size, hipMemcpyHostToDevice));
            hipEventRecord(stop,0);
            hipEventSynchronize(stop);
            hipEventElapsedTime(&ms, start, stop);
            printf(" %dMB: OK | Time Elapsed: %.2f\n", byte_size, ms);
            hipFree(device_buffer);

            hipHostUnregister(host_buffer);
            numa_free(host_buffer);
        }
    }
    hipEventDestroy(start);
    hipEventDestroy(stop);

    printf("\n");
}

// Unified memory test (tests if ReBAR is working properly)
static void check_unified_memory(int device_id) {
    printf("=== Unified Memory Check ===\n");

    void *um_ptr;
    hipError_t err = hipMallocManaged(&um_ptr, 256 * 1024 * 1024);

    if (err == hipSuccess) {
        printf("Managed memory (256MB): Success\n");

        // Touch the memory to trigger allocation
        hipMemset(um_ptr, 0x42, 256 * 1024 * 1024);
        hipDeviceSynchronize();
        printf("Managed memory write: Success\n");

        CHECK_HIP(hipFree(um_ptr));
    } else {
        printf("Managed memory: Failed - %s\n", hipGetErrorString(err));
    }
    printf("\n");
}

typedef struct {
    void *host;
    void *device;
    size_t size;
    int iterations;
    int direction; // 0 = H2D, 1 = D2H
    double elapsed_ms;
} transfer_args_t;

static void *transfer_thread(void *arg) {
    transfer_args_t *a = (transfer_args_t *)arg;
    a->elapsed_ms = 0;

    double start = get_time_ms();
    for (int i = 0; i < a->iterations; i++) {
        if (a->direction == 0) {
            CHECK_HIP(hipMemcpy(a->device, a->host, a->size, hipMemcpyHostToDevice));
        } else {
            CHECK_HIP(hipMemcpy(a->host, a->device, a->size, hipMemcpyDeviceToHost));
        }
    }
    hipDeviceSynchronize();
    a->elapsed_ms = get_time_ms() - start;

    return NULL;
}

static void run_bidirectional_test(size_t size, int iterations, const char *size_str) {
    printf("=== Bidirectional (Concurrent H2D + D2H) ===\n");
    printf("Transfer size: %s, Iterations: %d\n\n", size_str, iterations);
    printf("%12s  %12s  %12s  %12s\n", "H2D (GB/s)", "D2H (GB/s)", "Total (GB/s)", "Efficiency");
    printf("%12s  %12s  %12s  %12s\n", "------", "------", "------", "------");

    void *h_pinned, *d_ptr;
    CHECK_HIP(hipHostMalloc(&h_pinned, size, hipHostMallocDefault));
    CHECK_HIP(hipMalloc(&d_ptr, size));

    // Warmup
    CHECK_HIP(hipMemcpy(d_ptr, h_pinned, size, hipMemcpyHostToDevice));
    hipDeviceSynchronize();

    // Run concurrent transfers
    double h2d_total = 0, d2h_total = 0;
    for (int iter = 0; iter < iterations; iter++) {
        hipMemset(h_pinned, iter, size);

        pthread_t t1, t2;
        transfer_args_t a1 = {h_pinned, d_ptr, size, 1, 0, 0};
        transfer_args_t a2 = {h_pinned, d_ptr, size, 1, 1, 0};

        pthread_create(&t1, NULL, transfer_thread, &a1);
        pthread_create(&t2, NULL, transfer_thread, &a2);

        pthread_join(t1, NULL);
        pthread_join(t2, NULL);

        h2d_total += a1.elapsed_ms;
        d2h_total += a2.elapsed_ms;
    }

    double h2d_time = h2d_total / iterations;
    double d2h_time = d2h_total / iterations;
    double h2d_bw = (size / (1024.0*1024*1024)) / (h2d_time / 1000.0);
    double d2h_bw = (size / (1024.0*1024*1024)) / (d2h_time / 1000.0);
    double total_bw = h2d_bw + d2h_bw;
    double efficiency = total_bw / 60.0 * 100.0;

    printf("%12.2f  %12.2f  %12.2f  %12.1f%%\n", h2d_bw, d2h_bw, total_bw, efficiency);

    CHECK_HIP(hipHostFree(h_pinned));
    CHECK_HIP(hipFree(d_ptr));
    printf("\n");
}

static void run_unidirectional_tests(int iterations) {
    const size_t max_size = 512 * 1024 * 1024;
    size_t sizes[] = {
        4*1024, 16*1024, 64*1024, 256*1024, 1*1024*1024,
        4*1024*1024, 16*1024*1024, 64*1024*1024, 256*1024*1024, 512*1024*1024
    };
    const char *size_labels[] = {
        "4KB", "16KB", "64KB", "256KB", "1MB",
        "4MB", "16MB", "64MB", "256MB", "512MB"
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("=== Host -> Device Bandwidth ===\n");
    printf("%12s  %12s  %12s\n", "Size", "Time (ms)", "Bandwidth (GB/s)");
    printf("%12s  %12s  %12s\n", "------", "------", "------");

    void *h_pinned, *d_ptr;
    CHECK_HIP(hipHostMalloc(&h_pinned, max_size, hipHostMallocDefault));
    CHECK_HIP(hipMalloc(&d_ptr, max_size));

    // Warmup
    CHECK_HIP(hipMemcpy(d_ptr, h_pinned, 1024*1024, hipMemcpyHostToDevice));
    hipDeviceSynchronize();

    for (int i = 0; i < num_sizes; i++) {
        size_t size = sizes[i];
        hipMemset(h_pinned, 0, size);

        double start = get_time_ms();
        for (int iter = 0; iter < iterations; iter++) {
            CHECK_HIP(hipMemcpy(d_ptr, h_pinned, size, hipMemcpyHostToDevice));
        }
        hipDeviceSynchronize();
        double elapsed = get_time_ms() - start;

        double avg_time = elapsed / iterations;
        double bw = (size / (1024.0*1024*1024)) / (avg_time / 1000.0);
        printf("%12s  %12.3f  %12.2f\n", size_labels[i], avg_time, bw);
    }

    printf("\n=== Device -> Host Bandwidth ===\n");
    printf("%12s  %12s  %12s\n", "Size", "Time (ms)", "Bandwidth (GB/s)");
    printf("%12s  %12s  %12s\n", "------", "------", "------");

    for (int i = 0; i < num_sizes; i++) {
        size_t size = sizes[i];

        double start = get_time_ms();
        for (int iter = 0; iter < iterations; iter++) {
            CHECK_HIP(hipMemcpy(h_pinned, d_ptr, size, hipMemcpyDeviceToHost));
        }
        hipDeviceSynchronize();
        double elapsed = get_time_ms() - start;

        double avg_time = elapsed / iterations;
        double bw = (size / (1024.0*1024*1024)) / (avg_time / 1000.0);
        printf("%12s  %12.3f  %12.2f\n", size_labels[i], avg_time, bw);
    }

    CHECK_HIP(hipHostFree(h_pinned));
    CHECK_HIP(hipFree(d_ptr));
    printf("\n");
}

int main(int argc, char *argv[]) {
    int iterations = (argc > 1) ? atoi(argv[1]) : 100;
    int device_id = (argc > 2) ? atoi(argv[2]) : 0;

    CHECK_HIP(hipSetDevice(device_id));

    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, device_id));
    printf("GPU: %s\n", prop.name);
    printf("=====================================\n\n");

    check_rebar(device_id);
    check_unified_memory(device_id);
    run_unidirectional_tests(iterations);
    run_bidirectional_test(16 * 1024 * 1024, iterations, "16MB");
    run_bidirectional_test(256 * 1024 * 1024, iterations, "256MB");

    printf("Done.\n");
    return 0;
}