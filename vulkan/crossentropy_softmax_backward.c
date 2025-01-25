/*
Kernels for crossentropy and softmax backward pass.

version 1 is a straight-forward port from CPU code to kernel, parallel over B, T, V
bazel run :crossentropy_softmax_backward -- 1
*/

#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void crossentropy_softmax_backward_cpu(float* dlogits,
                                       float* dlosses, float* probs, int* targets,
                                       int B, int T, int V) {
    // dlogits: (B, T, V), probs: (B, T, V), dlosses: (B, T), targets: (B, T)
    // backwards through both softmax and crossentropy
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dlogits_bt = dlogits + b * T * V + t * V;
            float* probs_bt = probs + b * T * V + t * V;
            float dloss = dlosses[b * T + t];
            int ix = targets[b * T + t];
            for (int v = 0; v < V; v++) {
                float p = probs_bt[v];
                float indicator = v == ix ? 1.0f : 0.0f;
                dlogits_bt[v] += (p - indicator) * dloss;
            }  // (1, 1, V)
        }  // (1, T, V)
    }  // (B, T, V)
}

// ----------------------------------------------------------------------------
// kernel version dispatch

void select_kernel(int kernel_num,
                   Context* context, Kernel* kernel,
                   int B, int T, int V,
                   const int block_size) {
    switch (kernel_num) {
    case 1:
        append_shader(context, kernel, "shaders/crossentropy_softmax_backward_shader1.spv",
                      (Group){CEIL_DIV(B * T * V, block_size), 1, 1},
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

    const uint32_t B = 8;
    const uint32_t T = 1024;
    const uint32_t V = 50257;

    const uint32_t probs_size = B * T * V * sizeof(float);
    const uint32_t targets_size = B * T * sizeof(int);
    const uint32_t dlosses_size = B * T * sizeof(int);
    const uint32_t dlogits_size = B * T * V * sizeof(float);

    uint32_t sizes[] = { probs_size, targets_size, dlosses_size, dlogits_size };
    uint32_t num_tensors = sizeof(sizes) / sizeof(uint32_t);
    uint32_t offsets[num_tensors];
    for (int i = 0; i < num_tensors; i++) {
        offsets[i] = i == 0 ? 0 : offsets[i - 1] + sizes[i - 1];
    }

    Memory memory;
    allocate_memory(&context, &memory, num_tensors, sizes);

    float* probs = make_random_float_unit(probs_size / sizeof(float));
    int* targets = make_random_int(targets_size / sizeof(int), V);
    float* dlosses = make_random_float(dlosses_size / sizeof(float));
    float* dlogits = (float*)malloc(dlogits_size);
    memset(dlogits, 0, dlogits_size);

    float* d_probs;
    int* d_targets;
    float* d_dlosses;
    float* d_dlogits;

    vkMapMemory(context.device.logical_device, memory.heap, offsets[0], sizes[0], 0, (void**)&d_probs);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[1], sizes[1], 0, (void**)&d_targets);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[2], sizes[2], 0, (void**)&d_dlosses);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[3], sizes[3], 0, (void**)&d_dlogits);
    memcpy(d_probs, probs, probs_size);
    memcpy(d_targets, targets, targets_size);
    memcpy(d_dlosses, dlosses, dlosses_size);
    memset(d_dlogits, 0, dlogits_size);

    Kernel kernel;
    init_kernel(&context, &memory, &kernel);

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    select_kernel(kernel_num, &context, &kernel, B, T, V, 256);
    printf("Using kernel %d\n", kernel_num);

    Launcher launcher;
    init_launcher(&context, &launcher);

    uint32_t shapes[] = { B, T, V };
    launch_kernel(&context, &memory, &kernel, &launcher, 3, shapes, false);

    // first check the correctness of the kernel
    crossentropy_softmax_backward_cpu(dlogits, dlosses, probs, targets, B, T, V);
    float* dlogits_gpu = (float*)malloc(dlogits_size);
    memcpy(dlogits_gpu, d_dlogits, dlogits_size);
    for (int i = 0; i < B * T * V; i++) {
        // print the first few comparisons
        if (i < 10) {
            printf("%f %f\n", dlogits[i], dlogits_gpu[i]);
        }
        // ensure correctness for all elements
        if (fabs(dlogits[i] - dlogits_gpu[i]) > 1e-5) {
            printf("Mismatch at %d: %f vs %f\n", i, dlogits[i], dlogits_gpu[i]);
            exit(1);
        }
    }
    printf("Results match at block_size=256!\n");

    // time the kernel at different block sizes
    int block_sizes[] = { 32, 64, 128, 256, 512, 1024 };
    int repeat_times = 100;

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        destroy_kernel(&context, &kernel);
        init_kernel(&context, &memory, &kernel);
        select_kernel(kernel_num, &context, &kernel, B, T, V, block_size);

        uint64_t elapsed_ns = 0;
        uint64_t times[2];
        for (int i = 0; i < repeat_times; i++) {
            memset(d_dlogits, 0, dlogits_size);
            launch_kernel(&context, &memory, &kernel, &launcher, 3, shapes, i == 0);
            vkGetQueryPoolResults(context.device.logical_device, launcher.query_pool, 0, 2, 2 * sizeof(uint64_t), times, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
            elapsed_ns += times[1] - times[0];
        }
        float elapsed_time = (float)elapsed_ns / 1e6;

        printf("block_size %4d | time %f ms\n", block_size, elapsed_time / repeat_times);
    }

    vkUnmapMemory(context.device.logical_device, memory.heap);
    d_probs = d_dlosses = d_dlogits = NULL;
    d_targets = NULL;

    destroy_launcher(&context, &launcher);
    destroy_kernel(&context, &kernel);
    free_memory(&context, &memory);
    destroy_context(&context);

    free(probs);
    free(targets);
    free(dlogits);
    free(dlosses);
    free(dlogits_gpu);

    return 0;
}
