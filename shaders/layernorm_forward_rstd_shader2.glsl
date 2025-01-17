// naive port from layernorm_forward.cu kernel 2
#version 450
#pragma shader_stage(compute)

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

layout(set = 0, binding = 2) buffer writeonly rstdBuffer {
    float rstd[];
};

layout(set = 0, binding = 1) buffer readonly meanBuffer {
    float mean[];
};

layout(set = 0, binding = 3) buffer readonly inputBuffer {
    float inp[];
};

layout(push_constant) uniform args {
    uint B;
    uint T;
    uint C;
};

shared float partial_sum[gl_WorkGroupSize.x];

void main() {
    const uint idx = gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;
    const uint wg_size = gl_WorkGroupSize.x;
    const uint offset = idx * C;

    float m = mean[idx];
    float sum = 0.0f;
    for (uint c = tid; c < C; c += wg_size) {
        float diff = inp[offset + c] - m;
        sum += diff * diff;
    }
    partial_sum[tid] = sum;
    barrier();

    for (uint stride = wg_size / 2; stride >= 1; stride /= 2) {
        barrier();
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
    }

    if (tid == 0) {
        rstd[idx] = 1.0f / sqrt(partial_sum[0] / C + 1e-5f);
    }
}
