#version 450
#pragma shader_stage(compute)

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

layout(set = 0, binding = 1) buffer writeonly b1 {
    float dweight[];  // (O, C)
};

layout(set = 0, binding = 3) buffer readonly b3 {
    float dout[];  // (B, T, O)
};

layout(set = 0, binding = 4) buffer readonly b4 {
    float inp[];  // (B, T, C)
};

layout(push_constant) uniform args {
    uint B;
    uint T;
    uint C;
    uint O;
};

void main() {
    const uint BT = B * T;
    const uint o = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    const uint c = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;

    if (o < O && c < C) {
        for (uint bt = 0; bt < BT; bt++) {
            uint b = bt / BT;
            uint t = bt % BT;
            uint BTO_bt = b * BT * O + t * O;
            uint BTC_bt = b * BT * C + t * C;

            dweight[o * C + c] += inp[BTC_bt + c] * dout[BTO_bt + o];
        }
    }
}
