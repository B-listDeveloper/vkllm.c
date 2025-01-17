// naive port from layernorm_forward.cu kernel 2
#version 450
#pragma shader_stage(compute)

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

layout(set = 0, binding = 0) buffer writeonly outputBuffer {
    float outs[];
};

layout(set = 0, binding = 1) buffer readonly meanBuffer {
    float mean[];
};

layout(set = 0, binding = 2) buffer readonly rstdBuffer {
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
    uint B;
    uint T;
    uint C;
};

void main() {
    const uint idx = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    const uint bt = idx / C;
    const uint c = idx % C;

    float m = mean[bt];
    float s = rstd[bt];
    float xi = inp[idx];
    float n = s * (xi - m);
    float o = n * weight[c] + bias[c];

    outs[idx] = o;
}
