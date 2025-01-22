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

    if (idx < B * T) {
        const uint b = idx / T;
        const uint t = idx % T;
        const uint out_bt_offset = b * T * C + t * C;
        const int ix = inp[b * T + t];

        for (int c = 0; c < C; c++) {
            outs[out_bt_offset + c] = wte[ix * C + c] + wpe[t * C + c];
        }
    }
}
