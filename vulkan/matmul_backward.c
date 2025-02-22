/*
bazel run :matmul_forward -- 1
*/

#include <omp.h>

#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void matmul_backward_cpu(float* dinp, float* dweight, float* dbias,
                         float* dout, float* inp, float* weight,
                         int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B, T
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * OC + t * OC;
            float* dinp_bt = dinp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float* wrow = weight + o * C;
                for (int c = 0; c < C; c++) {
                    dinp_bt[c] += wrow[c] * dout_bt[o];
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #pragma omp parallel for
    for (int o = 0; o < OC; o++) {
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float* dout_bt = dout + b * T * OC + t * OC;
                float* inp_bt = inp + b * T * C + t * C;
                float* dwrow = dweight + o * C;
                if (dbias != NULL) { dbias[o] += dout_bt[o]; }
                for (int c = 0; c < C; c++) {
                    dwrow[c] += inp_bt[c] * dout_bt[o];
                }
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
        append_shader(context, kernel, "shaders/matmul_backward_input_shader1.spv",
                      (Group){B * T / sqrt_block_size, C / sqrt_block_size, 1},
                      (Group){sqrt_block_size, sqrt_block_size, 1});
        append_shader(context, kernel, "shaders/matmul_backward_weight_shader1.spv",
                      (Group){OC / sqrt_block_size, C / sqrt_block_size, 1},
                      (Group){sqrt_block_size, sqrt_block_size, 1});
        append_shader(context, kernel, "shaders/matmul_backward_bias_shader1.spv",
                      (Group){OC, 1, 1},
                      (Group){sqrt_block_size * sqrt_block_size, 1, 1});
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

    const uint32_t dinp_size = B * T * C * sizeof(float);
    const uint32_t dweight_size = OC * C * sizeof(float);
    const uint32_t dbias_size = OC * sizeof(float);
    const uint32_t dout_size = B * T * OC * sizeof(float);
    const uint32_t inp_size = B * T * C * sizeof(float);
    const uint32_t weight_size = OC * C * sizeof(float);
    const uint32_t ones_size = OC * sizeof(float);

    uint32_t sizes[] = {
        dinp_size, dweight_size, dbias_size, dout_size,
        inp_size, weight_size, ones_size,
    };
    uint32_t num_tensors = sizeof(sizes) / sizeof(uint32_t);
    uint32_t offsets[num_tensors];
    for (int i = 0; i < num_tensors; i++) {
        offsets[i] = i == 0 ? 0 : offsets[i - 1] + sizes[i - 1];
    }

    Memory memory;
    allocate_memory(&context, &memory, num_tensors, sizes);

    float* dinp = make_zeros_float(dinp_size / sizeof(float));
    float* dweight = make_zeros_float(dweight_size / sizeof(float));
    float* dbias = make_zeros_float(dbias_size / sizeof(float));
    float* dout = make_random_float(dout_size / sizeof(float));
    float* inp = make_random_float(inp_size / sizeof(float));
    float* weight = make_random_float(weight_size / sizeof(float));
    float* ones = make_ones_float(ones_size / sizeof(float));

    float* d_dinp;
    float* d_dweight;
    float* d_dbias;
    float* d_dout;
    float* d_inp;
    float* d_weight;
    float* d_ones;

    vkMapMemory(context.device.logical_device, memory.heap, offsets[0], sizes[0], 0, (void**)&d_dinp);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[1], sizes[1], 0, (void**)&d_dweight);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[2], sizes[2], 0, (void**)&d_dbias);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[3], sizes[3], 0, (void**)&d_dout);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[4], sizes[4], 0, (void**)&d_inp);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[5], sizes[5], 0, (void**)&d_weight);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[6], sizes[6], 0, (void**)&d_ones);
    memcpy(d_dinp, dinp, dinp_size);
    memcpy(d_dweight, dweight, dweight_size);
    memcpy(d_dbias, dbias, dbias_size);
    memcpy(d_dout, dout, dout_size);
    memcpy(d_inp, inp, inp_size);
    memcpy(d_weight, weight, weight_size);
    memcpy(d_ones, ones, ones_size);

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

    matmul_backward_cpu(dinp, dweight, dbias, dout, inp, weight, B, T, C, OC);
    float* dinp_gpu = (float*)malloc(dinp_size);
    memcpy(dinp_gpu, d_dinp, dinp_size);
    for (int i = 0; i < dinp_size / sizeof(float); i++) {
        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", dinp[i], dinp_gpu[i]);
        }
        // ensure correctness for all elements
        if (i >= 5 && fabsf(dinp[i] - dinp_gpu[i]) > 1e-3f) {
            printf("Mismatch at %d: %f vs %f\n", i, dinp[i], dinp_gpu[i]);
            exit(1);
        }
    }
    float* dweight_gpu = (float*)malloc(dweight_size);
    memcpy(dweight_gpu, d_dweight, dweight_size);
    for (int i = 0; i < dweight_size / sizeof(float); i++) {
        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", dweight[i], dweight_gpu[i]);
        }
        // ensure correctness for all elements
        if (i >= 5 && fabsf(dweight[i] - dweight_gpu[i]) > 1e-3f) {
            printf("Mismatch at %d: %f vs %f\n", i, dweight[i], dweight_gpu[i]);
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
            memset(d_dinp, 0, dinp_size);
            memset(d_dweight, 0, dweight_size);
            memset(d_dbias, 0, dbias_size);
            launch_kernel(&context, &memory, &kernel, &launcher, 4, shapes, i == 0);
            vkGetQueryPoolResults(context.device.logical_device, launcher.query_pool, 0, 2, 2 * sizeof(uint64_t), times, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
            elapsed_ns += times[1] - times[0];
        }
        float elapsed_time = (float)elapsed_ns / 1e6;

        // double-check if the result is correct
        memcpy(dinp_gpu, d_dinp, dinp_size);
        for (int i = 0; i < dinp_size / sizeof(float); i++) {
            if (fabsf(dinp[i] - dinp_gpu[i]) > 1e-4) {
                printf("Mismatch at %d: %f vs %f\n", i, dinp[i], dinp_gpu[i]);
                exit(EXIT_FAILURE);
            }
        }

        // napkin math: estimate the flops achieved
        // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
        float tflops = (float)B * T * C * OC * 2 * repeat_times / elapsed_time * 1e3f / 1e12f;
        printf("sqrt_block_size %4d | time %f ms | tflops %f\n", sqrt_block_size, elapsed_time, tflops);
    }

    vkUnmapMemory(context.device.logical_device, memory.heap);
    d_dinp = d_dweight = d_dbias = d_dout = d_inp = d_weight = d_ones = NULL;

    destroy_launcher(&context, &launcher);
    destroy_kernel(&context, &kernel);
    free_memory(&context, &memory);
    destroy_context(&context);

    free(dinp);
    free(dweight);
    free(dbias);
    free(dout);
    free(inp);
    free(weight);
    free(ones);
    free(dinp_gpu);
    free(dweight_gpu);

    return 0;
}
