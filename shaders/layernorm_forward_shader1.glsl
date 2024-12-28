// naive port from layernorm_forward.cu kernel 1
#version 450
#pragma shader_stage(compute)

layout(local_size_x_id = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer writeonly outputBuffer {
    float outs[];
};

layout(set = 0, binding = 1) buffer writeonly meanBuffer {
    float mean[];
};

layout(set = 0, binding = 2) buffer writeonly rstdBuffer {
    float rstd[];
};

layout(set = 0, binding = 3) buffer readonly inputBuffer {
    float inp[];
};

layout(set = 0, binding = 4) buffer readonly weightBuffer {
    float weight[];
};

layout(set = 0, binding = 5) buffer readonly biasBuffer {
    float bias[];
};

layout(push_constant) uniform args {
    uint N;
    uint C;
};

void main() {
    const uint idx = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    const float eps = 1e-5;

    if (idx < N) {
        const uint offset = idx * C;
        float m = 0.0f;
        for (uint c = 0; c < C; c++) {
            m += inp[offset + c];
        }
        m = m / C;

        float v = 0.0f;
        for (uint c = 0; c < C; c++) {
            float xshift = inp[offset + c] - m;
            v += xshift * xshift;
        }
        v = v / C;

        float s = 1.0f / sqrt(v + eps);

        for (uint c = 0; c < C; c++) {
            float n = (s * (inp[offset + c] - m));
            float o = n * weight[c] + bias[c];
            outs[offset + c] = o;
        }
        mean[idx] = m;
        rstd[idx] = s;
    }
}
