#version 450
#pragma shader_stage(compute)

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

layout(set = 0, binding = 0) buffer b0 {
    float outs[];
};

layout(set = 0, binding = 1) buffer readonly b1 {
    float inp[];
};

layout(push_constant) uniform args {
    uint N;
    uint C;
};

shared float shmem[gl_WorkGroupSize.x];

void main() {
    const uint idx = gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;
    const uint wg_size = gl_WorkGroupSize.x;
    const uint offset = idx * C;

    float maxval = -10000.0f;
    for (uint i = tid; i < C; i += wg_size) {
        maxval = max(maxval, inp[offset + i]);
    }
    shmem[tid] = maxval;
    barrier();

    // reductions
    for (uint stride = wg_size / 2; stride >= 1; stride /= 2) {
        barrier();
        if (tid < stride) {
            shmem[tid] = max(shmem[tid], shmem[tid + stride]);
        }
    }
    float channel_max = shmem[0];
    barrier();

    // compute expf and write the result to global memory
    for (uint i = tid; i < C; i += wg_size) {
        outs[offset + i] = exp(inp[offset + i] - channel_max);
    }
    barrier();

    // thread coarsening again, for the sum
    float sumval = 0.0f;
    for (uint i = tid; i < C; i += wg_size) {
        sumval += outs[offset + i];
    }
    shmem[tid] = sumval;
    barrier();

    // reductions
    for (uint stride = wg_size / 2; stride >= 1; stride /= 2) {
        barrier();
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
    }
    barrier();
    float sum = shmem[0];

    // divide the input values by the sum
    for (uint i = tid; i < C; i += wg_size) {
        outs[offset + i] /= sum;
    }
}
