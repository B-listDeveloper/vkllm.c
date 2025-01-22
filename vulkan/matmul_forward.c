/*
bazel run :matmul_forward -- 1
*/

#include <omp.h>

#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void matmul_forward_cpu(float* out,
                    float* inp, float* weight, float* bias,
                    int B, int T, int C, int OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                float* wrow = weight + o*C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// kernel version dispatch

void select_kernel(int kernel_num,
                   Context* context, Kernel* kernel,
                   int B, int T, int C, int OC,
                   const int sqrt_block_size) {
    switch (kernel_num) {
    case 1:
        append_shader(context, kernel, "shaders/matmul_forward_shader1.spv",
                      (Group){B * T / sqrt_block_size, OC / sqrt_block_size, 1},
                      (Group){sqrt_block_size, sqrt_block_size, 1});
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
    const int32_t OC = 768 * 4;  // expansion of 4, e.g. in the MLP

    const uint32_t out_size = B * T * OC * sizeof(float);
    const uint32_t inp_size = B * T * C * sizeof(float);
    const uint32_t weight_size = OC * C * sizeof(float);
    const uint32_t bias_size = OC * sizeof(float);

    uint32_t sizes[] = { out_size, inp_size, weight_size, bias_size };
    uint32_t num_tensors = sizeof(sizes) / sizeof(uint32_t);
    uint32_t offsets[num_tensors];
    for (int i = 0; i < num_tensors; i++) {
        offsets[i] = i == 0 ? 0 : offsets[i - 1] + sizes[i - 1];
    }

    Memory memory;
    allocate_memory(&context, &memory, num_tensors, sizes);

    float* out = (float*)malloc(out_size);
    float* inp = make_random_float(inp_size / sizeof(float));
    float* weight = make_random_float(weight_size / sizeof(float));
    float* bias = make_random_float(bias_size / sizeof(float));

    float* d_out;
    float* d_inp;
    float* d_weight;
    float* d_bias;

    vkMapMemory(context.device.logical_device, memory.heap, offsets[0], sizes[0], 0, (void**)&d_out);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[1], sizes[1], 0, (void**)&d_inp);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[2], sizes[2], 0, (void**)&d_weight);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[3], sizes[3], 0, (void**)&d_bias);
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
    select_kernel(kernel_num, &context, &kernel, B, T, C, OC, 16);
    printf("Using kernel %d\n", kernel_num);

    Launcher launcher;
    init_launcher(&context, &launcher);

    uint32_t shapes[] = { B, T, C, OC };
    launch_kernel(&context, &memory, &kernel, &launcher, 4, shapes, false);

    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);
    float* out_gpu = (float*)malloc(out_size);
    memcpy(out_gpu, d_out, out_size);
    for (int i = 0; i < B * T * OC; i++) {
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
    int sqrt_block_sizes[] = { 4, 8, 16, 32 };
    int repeat_times = 10;

    for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
        int sqrt_block_size = sqrt_block_sizes[j];

        destroy_kernel(&context, &kernel);
        init_kernel(&context, &memory, &kernel);
        select_kernel(kernel_num, &context, &kernel, B, T, C, OC, sqrt_block_size);

        uint64_t elapsed_ns = 0;
        uint64_t times[2];
        for (int i = 0; i < repeat_times; i++) {
            launch_kernel(&context, &memory, &kernel, &launcher, 4, shapes, i == 0);
            vkGetQueryPoolResults(context.device.logical_device, launcher.query_pool, 0, 2, 2 * sizeof(uint64_t), times, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
            elapsed_ns += times[1] - times[0];
        }
        float elapsed_time = (float)elapsed_ns / 1e6;

        // double-check if the result is correct
        memcpy(out_gpu, d_out, out_size);
        for (int i = 0; i < B * T * OC; i++) {
            if (fabsf(out[i] - out_gpu[i]) > 1e-4) {
                printf("Mismatch at %d: %f vs %f\n", i, out[i], out_gpu[i]);
                exit(EXIT_FAILURE);
            }
        }

        // napkin math: estimate the flops achieved
        // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
        float tflops = (float)B * T * C * OC * 2 * repeat_times / elapsed_time * 1e3f / 1e12f;
        printf("sqrt_block_size %4d | time %f ms | tflops %f\n", sqrt_block_size, elapsed_time, tflops);
    }

    vkUnmapMemory(context.device.logical_device, memory.heap);
    d_out = d_inp = d_weight = d_bias = NULL;

    destroy_launcher(&context, &launcher);
    destroy_kernel(&context, &kernel);
    free_memory(&context, &memory);
    destroy_context(&context);

    free(out);
    free(inp);
    free(weight);
    free(bias);
    free(out_gpu);

    return 0;
}
