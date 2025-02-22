#version 450
#pragma shader_stage(compute)

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

layout(set = 0, binding = 0) buffer writeonly b0 {
    float dinp[];  // (B, T, C)
};

layout(set = 0, binding = 3) buffer readonly b3 {
    float dout[];  // (B, T, O)
};

layout(set = 0, binding = 5) buffer readonly b5 {
    float weight[];  // (O, C)
};

layout(push_constant) uniform args {
    uint B;
    uint T;
    uint C;
    uint O;
};

void main() {
    const uint BT = B * T;
    const uint bt = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    const uint c = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;

    if (bt < BT && c < C) {
        const uint b = bt / BT;
        const uint t = bt % BT;

        const uint BTO_bt = b * BT * O + t * O;
        const uint BTC_bt = b * BT * C + t * C;

        for (uint o = 0; o < O; o++) {
            dinp[BTC_bt + c] += weight[o * C + c] * dout[BTO_bt + o];
        }
    }
}
