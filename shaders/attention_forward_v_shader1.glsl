#version 450
#pragma shader_stage(compute)

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

layout(set = 0, binding = 0) buffer writeonly b0 {
    float outs[];
};

layout(set = 0, binding = 2) buffer readonly b2 {
    float att[];
};

layout(set = 0, binding = 3) buffer readonly b3 {
    float inp[];
};

layout(push_constant) uniform args {
    uint B;
    uint T;
    uint C;
    uint H;
};

void main() {
    const uint idx = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    if (idx < B * H * T) {
        const uint h = idx % H;
        const uint t = (idx / H) % T;
        const uint b = idx / (H * T);

        const uint C3 = 3 * C;
        const uint hs = C / H;  // head size

        const uint out_bt_offset = b * T * C + t * C + h * hs;
        const uint att_bht_offset = b * H * T * T + h * T * T + t * T;

        for (int i = 0; i < hs; i++) { outs[out_bt_offset + i] = 0.0f; }
        for (int t2 = 0; t2 <= t; t2++) {
            uint value_t2_offset = b * T * C3 + t2 * C3 + h * hs + 2 * C;  // + 2 * C because it's value
            for (int c = 0; c < hs; c++) {
                outs[out_bt_offset + c] += att[att_bht_offset + t2] * inp[value_t2_offset + c];
            }
        }
    }
}
