#version 450
#pragma shader_stage(compute)

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

layout(set = 0, binding = 0) buffer writeonly b0 {
    float outs[];
};

layout(set = 0, binding = 1) buffer readonly b1 {
    int inp[];
};

layout(set = 0, binding = 2) buffer readonly b2 {
    float wte[];
};

layout(set = 0, binding = 3) buffer readonly b3 {
    float wpe[];
};

layout(push_constant) uniform args {
    uint B;
    uint T;
    uint C;
};

void main() {
    const uint idx = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    if (idx < B * T * C) {
        const uint bt = idx / C;
        const uint b = bt / T;
        const uint t = bt % T;
        const uint c = idx % C;

        const int ix = inp[b * T + t];
        outs[b * T * C + t * C + c] = wte[ix * C + c] + wpe[t * C + c];
    }
}
