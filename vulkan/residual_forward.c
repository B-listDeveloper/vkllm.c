/*
Kernels for attention forward pass.

version 1 is naive port from CPU code to kernel
bazel run :residual_forward -- 1
*/

#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void residual_forward_cpu(float* out, float* inp1, float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

// ----------------------------------------------------------------------------
// kernel version dispatch

void select_kernel(int kernel_num,
                   Context* context, Kernel* kernel,
                   int N,
                   const int block_size) {
    switch (kernel_num) {
    case 1:
        append_shader(context, kernel, "shaders/residual_forward_shader1.spv",
                      (Group){CEIL_DIV(N, block_size), 1, 1},
                      (Group){block_size, 1, 1});
        break;
    default:
        printf("Invalid kernel number\n");
        exit(EXIT_FAILURE);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
    srand(0);

    Context context;
    init_context(&context);

    // memory allocation
    const uint32_t B = 8;
    const uint32_t T = 1024;
    const uint32_t C = 768;

    const uint32_t out_size = B * T * C * sizeof(float);
    const uint32_t inp1_size = B * T * C * sizeof(float);
    const uint32_t inp2_size = B * T * C * sizeof(float);

    uint32_t sizes[] = { out_size, inp1_size, inp2_size };
    uint32_t num_tensors = sizeof(sizes) / sizeof(uint32_t);
    uint32_t offsets[num_tensors];
    for (int i = 0; i < num_tensors; i++) {
        offsets[i] = i == 0 ? 0 : offsets[i - 1] + sizes[i - 1];
    }

    Memory memory;
    allocate_memory(&context, &memory, num_tensors, sizes);

    float* out = (float*)malloc(out_size);
    float* inp1 = make_random_float(inp1_size / sizeof(float));
    float* inp2 = make_random_float(inp2_size / sizeof(float));

    float* d_out;
    float* d_inp1;
    float* d_inp2;

    vkMapMemory(context.device.logical_device, memory.heap, offsets[0], sizes[0], 0, (void**)&d_out);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[1], sizes[1], 0, (void**)&d_inp1);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[2], sizes[2], 0, (void**)&d_inp2);
    memcpy(d_inp1, inp1, inp1_size);
    memcpy(d_inp2, inp2, inp2_size);

    Kernel kernel;
    init_kernel(&context, &memory, &kernel);

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    select_kernel(kernel_num, &context, &kernel, B * T * C, 256);
    printf("Using kernel %d\n", kernel_num);

    Launcher launcher;
    init_launcher(&context, &launcher);

    uint32_t shapes[] = { B * T * C };
    launch_kernel(&context, &memory, &kernel, &launcher, 1, shapes, false);

    residual_forward_cpu(out, inp1, inp2, B * T * C);
    float* out_gpu = (float*)malloc(out_size);
    memcpy(out_gpu, d_out, out_size);
    for (int i = 0; i < B * T * C; i++) {
        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", out[i], out_gpu[i]);
        }
        // ensure correctness for all elements
        if (i >= 5 && fabsf(out[i] - out_gpu[i]) > 1e-4) {
            printf("Mismatch at %d: %f vs %f\n", i, out[i], out_gpu[i]);
            exit(1);
        }
    }
    printf("Results match at block_size=256!\n");

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    int repeat_times = 1000;

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        destroy_kernel(&context, &kernel);
        init_kernel(&context, &memory, &kernel);
        select_kernel(kernel_num, &context, &kernel, B * T * C, block_size);

        uint64_t elapsed_ns = 0;
        uint64_t times[2];
        for (int i = 0; i < repeat_times; i++) {
            launch_kernel(&context, &memory, &kernel, &launcher, 1, shapes, i == 0);
            vkGetQueryPoolResults(context.device.logical_device, launcher.query_pool, 0, 2, 2 * sizeof(uint64_t), times, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
            elapsed_ns += times[1] - times[0];
        }
        float elapsed_time = (float)elapsed_ns / 1e6;

        // double-check if the result is correct
        memcpy(out_gpu, d_out, out_size);
        for (int i = 0; i < B * T * C; i++) {
            if (fabsf(out[i] - out_gpu[i]) > 1e-4) {
                printf("Mismatch at %d: %f vs %f\n", i, out[i], out_gpu[i]);
                exit(EXIT_FAILURE);
            }
        }

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 2 read and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 3 * 4;
        float memory_bandwidth = memory_ops / (elapsed_time / repeat_times) / 1e6;

        printf("block_size %4d | time %f ms | bandwidth %f GB/s\n", block_size, elapsed_time / repeat_times, memory_bandwidth);
    }

    vkUnmapMemory(context.device.logical_device, memory.heap);
    d_out = d_inp1 = d_inp2 = NULL;

    destroy_launcher(&context, &launcher);
    destroy_kernel(&context, &kernel);
    free_memory(&context, &memory);
    destroy_context(&context);

    free(out);
    free(inp1);
    free(inp2);
    free(out_gpu);

    return 0;
}
