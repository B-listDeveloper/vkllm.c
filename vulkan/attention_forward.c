/*
Kernels for attention forward pass.

version 1 is naive port from CPU code to kernel, parallelize over batch, time, heads only
bazel run :attention_forward -- 1
*/

#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void attention_forward_cpu(float* out, float* preatt, float* att,
                           float* inp,
                           int B, int T, int C, int H) {
    // input is (B, T, 3C) Q, K, V
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int C3 = 3 * C;
    int hs = C / H;  // head size
    float scale = 1.0 / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < H; h++) {
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bht = preatt + b * H * T * T + h * T * T + t * T;
                float* att_bht = att + b * H * T * T + h * T * T + t * T;

                // pass 1: calculate query dot key and maxval
                float maxval = -INFINITY;
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;  // + C because it's key

                    // (query) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bht[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                //   => the denominator of softmax
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bht[t2] - maxval);
                    expsum += expv;
                    att_bht[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bht[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bht[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                float* out_bt = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bt[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + 2 * C;  // + 2 * C because it's value
                    for (int c = 0; c < hs; c++) {
                        out_bt[c] += att_bht[t2] * value_t2[c];
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// kernel version dispatch

void select_kernel(int kernel_num,
                   Context* context, Kernel* kernel,
                   int B, int T, int C, int H,
                   const int block_size) {
    switch (kernel_num) {
    case 1:
        append_shader(context, kernel, "shaders/attention_forward_qk_shader1.spv",
                      (Group){CEIL_DIV(B * H * T * T, block_size), 1, 1},
                      (Group){block_size, 1, 1});
        append_shader(context, kernel, "shaders/attention_forward_softmax_shader1.spv",
                      (Group){CEIL_DIV(B * T * H, block_size), 1, 1},
                      (Group){block_size, 1, 1});
        append_shader(context, kernel, "shaders/attention_forward_v_shader1.spv",
                      (Group){CEIL_DIV(B * T * H, block_size), 1, 1},
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

    // create host memory of random numbers
    const uint32_t B = 8;
    const uint32_t T = 1024;
    const uint32_t C = 768;
    const uint32_t H = 12;

    const uint32_t out_size = B * T * C * sizeof(float);
    const uint32_t preatt_size = B * H * T * T * sizeof(float);
    const uint32_t att_size = B * H * T * T * sizeof(float);
    const uint32_t inp_size = B * T * (3 * C) * sizeof(float);

    uint32_t sizes[] = { out_size, preatt_size, att_size, inp_size };
    uint32_t num_tensors = sizeof(sizes) / sizeof(uint32_t);
    uint32_t offsets[num_tensors];
    for (int i = 0; i < num_tensors; i++) {
        offsets[i] = i == 0 ? 0 : offsets[i - 1] + sizes[i - 1];
    }

    Memory memory;
    allocate_memory(&context, &memory, num_tensors, sizes);

    float* out = (float*)malloc(out_size);
    float* preatt = (float*)malloc(preatt_size);
    float* att = (float*)malloc(att_size);
    float* inp = make_random_float(inp_size / sizeof(float));

    float* d_out;
    float* d_preatt;
    float* d_att;
    float* d_inp;

    vkMapMemory(context.device.logical_device, memory.heap, offsets[0], sizes[0], 0, (void**)&d_out);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[1], sizes[1], 0, (void**)&d_preatt);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[2], sizes[2], 0, (void**)&d_att);
    vkMapMemory(context.device.logical_device, memory.heap, offsets[3], sizes[3], 0, (void**)&d_inp);
    memcpy(d_out, out, out_size);
    memcpy(d_preatt, preatt, preatt_size);
    memcpy(d_att, att, att_size);
    memcpy(d_inp, inp, inp_size);

    Kernel kernel;
    init_kernel(&context, &memory, &kernel);

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    select_kernel(kernel_num, &context, &kernel, B, T, C, H, 256);
    printf("Using kernel %d\n", kernel_num);

    Launcher launcher;
    init_launcher(&context, &launcher);

    uint32_t shapes[] = { B, T, C, H };
    launch_kernel(&context, &memory, &kernel, &launcher, 4, shapes, false);

    attention_forward_cpu(out, preatt, att, inp, B, T, C, H);
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
    int block_sizes[] = { 32, 64, 128, 256, 512 };
    int repeat_times = 10;

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        destroy_kernel(&context, &kernel);
        init_kernel(&context, &memory, &kernel);
        select_kernel(kernel_num, &context, &kernel, B, T, C, H, block_size);

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
        for (int i = 0; i < B * T * C; i++) {
            if (fabsf(out[i] - out_gpu[i]) > 1e-4) {
                printf("Mismatch at %d: %f vs %f\n", i, out[i], out_gpu[i]);
                exit(EXIT_FAILURE);
            }
        }

        printf("block_size %4d | time %f ms\n", block_size, elapsed_time);
    }

    vkUnmapMemory(context.device.logical_device, memory.heap);
    d_out = d_preatt = d_att = d_inp = NULL;

    destroy_launcher(&context, &launcher);
    destroy_kernel(&context, &kernel);
    free_memory(&context, &memory);
    destroy_context(&context);

    free(out);
    free(preatt);
    free(att);
    free(inp);
    free(out_gpu);

    return 0;
}
