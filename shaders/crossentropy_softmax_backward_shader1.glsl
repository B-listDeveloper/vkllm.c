#version 450
#pragma shader_stage(compute)

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

layout(set = 0, binding = 0) buffer readonly b0 {
    float probs[];  // (B, T, V)
};

layout(set = 0, binding = 1) buffer readonly b1 {
    int targets[];  // (B, T)
};

layout(set = 0, binding = 2) buffer readonly b2 {
    float dlosses[];  // (B, T)
};

layout(set = 0, binding = 3) buffer writeonly b3 {
    float dlogits[];  // (B, T, V)
};

layout(push_constant) uniform args {
    uint B;
    uint T;
    uint V;
};

void main() {
    const uint idx = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    if (idx < B * T * V) {
        const uint v = idx % V;
        const uint t = (idx / V) % T;
        const uint b = idx / (T * V);

        const uint BT_bt = b * T + t;
        const uint BTV_bt = BT_bt * V;

        float indicator = v == targets[BT_bt] ? 1.0f : 0.0f;
        dlogits[BTV_bt + v] += (probs[BTV_bt + v] - indicator) * dlosses[BT_bt];
    }
}
