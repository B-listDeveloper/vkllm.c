#version 450
#pragma shader_stage(compute)

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

layout(set = 0, binding = 0) buffer b0 {
    float losses[];
};

layout(set = 0, binding = 1) buffer readonly b1 {
    float probs[];
};

layout(set = 0, binding = 2) buffer readonly b2 {
    int targets[];
};

layout(push_constant) uniform args {
    uint B;
    uint T;
    uint V;
};

void main() {
    const uint idx = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    if (idx < B * T) {
        const uint b = idx / T;
        const uint t = idx % T;
        const uint targets_bt_offset = b * T + t;
        const uint probs_bt_offset = b * T * V + t * V;

        int ix = targets[targets_bt_offset];
        losses[targets_bt_offset] = -log(probs[probs_bt_offset + ix]);
    }
}
