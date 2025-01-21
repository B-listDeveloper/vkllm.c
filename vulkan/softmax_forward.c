/*
Kernels for softmax forward pass.

version 1 is naive port from CPU code to kernel: parallelizes over B, T, loops over C
bazel run softmax_forward -- 1

version 2 is a fused kernel that parallelizes over all of B, T, C
bazel run softmax_forward -- 2
*/

#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void softmax_forward_cpu(float* out, float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    for (int i = 0; i < N; i++) {
        float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }
        float sum = 0.0f;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        for (int j = 0; j < C; j++) {
            out_row[j] /= sum;
        }
    }
}

// ----------------------------------------------------------------------------
// kernel version dispatch

void select_kernel(int kernel_num,
                   Context* context, Kernel* kernel,
                   int N, int C,
                   const int block_size) {
    switch (kernel_num) {
    case 1:
        append_shader(context, kernel, "shaders/softmax_forward_shader1.spv",
                      (Group){CEIL_DIV(N, block_size), 1, 1},
                      (Group){block_size, 1, 1});
        break;
    case 2:
        append_shader(context, kernel, "shaders/softmax_forward_shader2.spv",
                      (Group){N, 1, 1},
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

    const uint32_t out_size = B * T * T * sizeof(float);
    const uint32_t inp_size = B * T * T * sizeof(float);

    uint32_t sizes[] = { out_size, inp_size };
    uint32_t num_tensors = sizeof(sizes) / sizeof(uint32_t);
    uint32_t offsets[num_tensors];
    for (int i = 0; i < num_tensors; i++) {
        offsets[i] = i == 0 ? 0 : offsets[i - 1] + sizes[i - 1];
    }

    Memory memory;
    allocate_memory(&context, &memory, num_tensors, sizes);

    float* out = (float*)malloc(out_size);
    float* inp = make_random_float(inp_size / sizeof(float));

    float* d_out;
    float* d_inp;

    vkMapMemory(context.device.logical_device, memory.heap, offsets[0], sizes[0], 0, (void**)&d_out);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[1], sizes[1], 0, (void**)&d_inp);
    memcpy(d_out, out, out_size);
    memcpy(d_inp, inp, inp_size);

    Kernel kernel;
    init_kernel(&context, &memory, &kernel);

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    select_kernel(kernel_num, &context, &kernel, B * T, T, 256);
    printf("Using kernel %d\n", kernel_num);

    Launcher launcher;
    init_launcher(&context, &launcher);

    uint32_t shapes[] = { B * T, T };
    launch_kernel(&context, &memory, &kernel, &launcher, 2, shapes, false);

    softmax_forward_cpu(out, inp, B * T, T);
    float* out_gpu = (float*)malloc(out_size);
    memcpy(out_gpu, d_out, out_size);
    for (int i = 0; i < B * T * T; i++) {
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
    int block_sizes[] = { 32, 64, 128, 256, 512, 1024 };
    int repeat_times = 1000;

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        destroy_kernel(&context, &kernel);
        init_kernel(&context, &memory, &kernel);
        select_kernel(kernel_num, &context, &kernel, B * T, T, block_size);

        uint64_t elapsed_ns = 0;
        uint64_t times[2];
        for (int i = 0; i < repeat_times; i++) {
            launch_kernel(&context, &memory, &kernel, &launcher, 2, shapes, i == 0);
            vkGetQueryPoolResults(context.device.logical_device, launcher.query_pool, 0, 2, 2 * sizeof(uint64_t), times, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
            elapsed_ns += times[1] - times[0];
        }
        float elapsed_time = (float)elapsed_ns / 1e6;

        // double-check if the result is correct
        memcpy(out_gpu, d_out, out_size);
        for (int i = 0; i < B * T * T; i++) {
            if (fabsf(out[i] - out_gpu[i]) > 1e-4) {
                printf("Mismatch at %d: %f vs %f\n", i, out[i], out_gpu[i]);
                exit(EXIT_FAILURE);
            }
        }

        printf("block_size %4d | time %f ms\n", block_size, elapsed_time / repeat_times);
    }

    free(out);
    free(inp);
    free(out_gpu);

    return 0;
}