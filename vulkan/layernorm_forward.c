/*
version 1 is naive port from CPU code to shader: parallelizes over B, T, loops over C
bazel run :layernorm_forward -- 1

version 2 parallelizes over all of B,T,C
bazel run :layernorm_forward -- 2
*/

#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 layernorm forward pass
void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                           float* inp, float* weight, float* bias,
                           int B, int T, int C) {
    // out: (B, T, C), mean: (B, T), rstd: (B, T), inp: (B, T, C), weight: (C), bias: (C)
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b, t, :]
            float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int c = 0; c < C; c++) {
                m += x[c];
            }
            m = m / C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int c = 0; c < C; c++) {
                float xshift = x[c] - m;
                v += xshift * xshift;
            }
            v = v / C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b, t, :]
            float* out_bt = out + b * T * C + t * C;
            for (int c = 0; c < C; c++) {
                float n = ((x[c] - m) * s);  // normalized output
                float o = n * weight[c] + bias[c];  // scale and shift it
                out_bt[c] = o;  // write
            }  // (1, 1, C)
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }  // (1, T, C)
    }  // (B, T, C)
}

// ----------------------------------------------------------------------------
// kernel version dispatch

void select_kernel(int kernel_num,
                   Context* context, Kernel* kernel,
                   int B, int T, int C,
                   const int block_size) {
    switch (kernel_num) {
    case 1:
        append_shader(context, kernel, "shaders/layernorm_forward_shader1.spv",
                      (Group){CEIL_DIV(B * T, block_size), 1, 1},
                      (Group){block_size, 1, 1});
        break;
    case 2:
        append_shader(context, kernel, "shaders/layernorm_forward_mean_shader2.spv",
                      (Group){B * T, 1, 1},
                      (Group){block_size, 1, 1});
        append_shader(context, kernel, "shaders/layernorm_forward_rstd_shader2.spv",
                      (Group){B * T, 1, 1},
                      (Group){block_size, 1, 1});
        append_shader(context, kernel, "shaders/layernorm_forward_norm_shader2.spv",
                      (Group){CEIL_DIV(B * T * C, 256), 1, 1},
                      (Group){256, 1, 1});
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
    const uint32_t mean_size = B * T * sizeof(float);
    const uint32_t rstd_size = B * T * sizeof(float);
    const uint32_t inp_size = B * T * C * sizeof(float);
    const uint32_t weight_size = C * sizeof(float);
    const uint32_t bias_size = C * sizeof(float);

    uint32_t sizes[] = { out_size, mean_size, rstd_size, inp_size, weight_size, bias_size };
    uint32_t num_tensors = sizeof(sizes) / sizeof(uint32_t);
    uint32_t offsets[num_tensors];
    for (int i = 0; i < num_tensors; i++) {
        offsets[i] = i == 0 ? 0 : offsets[i - 1] + sizes[i - 1];
    }

    Memory memory;
    allocate_memory(&context, &memory, num_tensors, sizes);

    // create host memory of random numbers
    float* out = (float*)malloc(out_size);
    float* mean = (float*)malloc(mean_size);
    float* rstd = (float*)malloc(rstd_size);
    float* inp = make_random_float(inp_size / sizeof(float));
    float* weight = make_random_float(weight_size / sizeof(float));
    float* bias = make_random_float(bias_size / sizeof(float));

    float* d_out;
    float* d_mean;
    float* d_rstd;
    float* d_inp;
    float* d_weight;
    float* d_bias;

    vkMapMemory(context.device.logical_device, memory.heap, offsets[0], sizes[0], 0, (void**)&d_out);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[1], sizes[1], 0, (void**)&d_mean);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[2], sizes[2], 0, (void**)&d_rstd);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[3], sizes[3], 0, (void**)&d_inp);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[4], sizes[4], 0, (void**)&d_weight);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[5], sizes[5], 0, (void**)&d_bias);
    memcpy(d_mean, mean, mean_size);
    memcpy(d_rstd, rstd, rstd_size);
    memcpy(d_inp, inp, inp_size);
    memcpy(d_weight, weight, weight_size);
    memcpy(d_bias, bias, bias_size);

    Kernel kernel;
    init_kernel(&context, &memory, &kernel);

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    select_kernel(kernel_num, &context, &kernel, B, T, C, 256);
    printf("Using kernel %d\n", kernel_num);

    Launcher launcher;
    init_launcher(&context, &launcher);

    uint32_t shapes[] = { B, T, C };
    launch_kernel(&context, &memory, &kernel, &launcher, 3, shapes, false);

    layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);
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

    // time the shader at different block sizes
    int block_sizes[] = { 32, 64, 128, 256, 512, 1024 };
    int repeat_times = 100;

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        destroy_kernel(&context, &kernel);
        init_kernel(&context, &memory, &kernel);
        select_kernel(kernel_num, &context, &kernel, B, T, C, block_size);

        uint64_t elapsed_ns = 0;
        uint64_t times[2];
        for (int i = 0; i < repeat_times; i++) {
            launch_kernel(&context, &memory, &kernel, &launcher, 3, shapes, i == 0);
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
        // e.g. A100 40GB PCIe is advertised at 1,555 GB/s
        long memory_ops = (2 * B * T * C) * 4;  // * 4 for float
        float memory_bandwidth = memory_ops / (elapsed_time / repeat_times) / 1e6;

        printf("block_size %4d | time %f ms | bandwidth %f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    destroy_launcher(&context, &launcher);
    destroy_kernel(&context, &kernel);
    free_memory(&context, &memory);
    destroy_context(&context);

    free(out);
    free(d_mean);
    free(d_rstd);
    free(inp);
    free(weight);
    free(bias);
    free(out_gpu);

    return 0;
}
