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

void main() {
    const uint idx = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    if (idx < N) {
        const uint offset = idx * C;
        float maxval = -10000.0f;
        for (int j = 0; j < C; j++) {
            if (inp[offset + j] > maxval) {
                maxval = inp[offset + j];
            }
        }

        float sum = 0.0f;
        for (int j = 0; j < C; j++) {
            outs[offset + j] = exp(inp[offset + j] - maxval);
            sum += outs[offset + j];
        }
        for (int j = 0; j < C; j++) {
            outs[offset + j] /= sum;
        }
    }
}
