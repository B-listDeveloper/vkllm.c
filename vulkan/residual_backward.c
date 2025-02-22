/*
bazel run :residual_backward -- 1
*/

#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void residual_backward_cpu(float* dinp1, float* dinp2, float* dout, int N) {
    for (int i = 0; i < N; i++) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
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
        append_shader(context, kernel, "shaders/residual_backward_shader1.spv",
                      (Group){N / block_size, 1, 1},
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
    const int32_t B = 8;
    const int32_t T = 1024;
    const int32_t C = 768;

    const uint32_t dinp1_size = B * T * C * sizeof(float);
    const uint32_t dinp2_size = B * T * C * sizeof(float);
    const uint32_t dout_size = B * T * C * sizeof(float);

    uint32_t sizes[] = { dinp1_size, dinp2_size, dout_size };
    uint32_t num_tensors = sizeof(sizes) / sizeof(uint32_t);
    uint32_t offsets[num_tensors];
    for (int i = 0; i < num_tensors; i++) {
        offsets[i] = i == 0 ? 0 : offsets[i - 1] + sizes[i - 1];
    }

    Memory memory;
    allocate_memory(&context, &memory, num_tensors, sizes);

    float* dinp1 = make_zeros_float(dinp1_size / sizeof(float));
    float* dinp2 = make_zeros_float(dinp2_size / sizeof(float));
    float* dout = make_random_float(dout_size / sizeof(float));

    float* d_dinp1;
    float* d_dinp2;
    float* d_dout;

    vkMapMemory(context.device.logical_device, memory.heap, offsets[0], sizes[0], 0, (void**)&d_dinp1);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[1], sizes[1], 0, (void**)&d_dinp2);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[2], sizes[2], 0, (void**)&d_dout);
    memcpy(d_dinp1, dinp1, dinp1_size);
    memcpy(d_dinp2, dinp2, dinp2_size);
    memcpy(d_dout, dout, dout_size);

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

    residual_backward_cpu(dinp1, dinp2, dout, B * T * C);
    float* dinp_gpu = (float*)malloc(dinp1_size);
    memcpy(dinp_gpu, d_dinp1, dinp1_size);
    for (int i = 0; i < B * T * C; i++) {
        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", dinp1[i], dinp_gpu[i]);
        }
        // ensure correctness for all elements
        if (i >= 5 && fabsf(dinp1[i] - dinp_gpu[i]) > 1e-4) {
            printf("Mismatch at %d: %f vs %f\n", i, dinp1[i], dinp_gpu[i]);
            exit(1);
        }
    }
    memcpy(dinp_gpu, d_dinp2, dinp2_size);
    for (int i = 0; i < B * T * C; i++) {
        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", dinp2[i], dinp_gpu[i]);
        }
        // ensure correctness for all elements
        if (i >= 5 && fabsf(dinp2[i] - dinp_gpu[i]) > 1e-4) {
            printf("Mismatch at %d: %f vs %f\n", i, dinp2[i], dinp_gpu[i]);
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
            memset(d_dinp1, 0, dinp1_size);
            memset(d_dinp2, 0, dinp1_size);
            launch_kernel(&context, &memory, &kernel, &launcher, 1, shapes, i == 0);
            vkGetQueryPoolResults(context.device.logical_device, launcher.query_pool, 0, 2, 2 * sizeof(uint64_t), times, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
            elapsed_ns += times[1] - times[0];
        }
        float elapsed_time = (float)elapsed_ns / 1e6;

        // double-check if the result is correct
        memcpy(dinp_gpu, d_dinp1, dinp1_size);
        for (int i = 0; i < B * T * C; i++) {
            if (fabsf(dinp1[i] - dinp_gpu[i]) > 1e-4) {
                printf("Mismatch at %d: %f vs %f\n", i, dinp1[i], dinp_gpu[i]);
                exit(EXIT_FAILURE);
            }
        }
        memcpy(dinp_gpu, d_dinp2, dinp2_size);
        for (int i = 0; i < B * T * C; i++) {
            if (fabsf(dinp2[i] - dinp_gpu[i]) > 1e-4) {
                printf("Mismatch at %d: %f vs %f\n", i, dinp2[i], dinp_gpu[i]);
                exit(EXIT_FAILURE);
            }
        }

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 1 read and 2 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 3 * 4;
        float memory_bandwidth = memory_ops / (elapsed_time / repeat_times) / 1e6;

        printf("block_size %4d | time %f ms | bandwidth %f GB/s\n", block_size, elapsed_time / repeat_times, memory_bandwidth);
    }

    vkUnmapMemory(context.device.logical_device, memory.heap);
    d_dinp1 = d_dinp2 = d_dout = NULL;

    destroy_launcher(&context, &launcher);
    destroy_kernel(&context, &kernel);
    free_memory(&context, &memory);
    destroy_context(&context);

    free(dinp1);
    free(dinp2);
    free(dout);
    free(dinp_gpu);

    return 0;
}
