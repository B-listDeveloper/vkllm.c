// naive port from matmul_forward.cu kernel 1
#version 450
#pragma shader_stage(compute)

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

layout(set = 0, binding = 0) buffer writeonly outputBuffer {
    float outs[];
};

layout(set = 0, binding = 1) buffer readonly inputBuffer {
    float inp[];
};

layout(set = 0, binding = 2) buffer readonly weightBuffer {
    float weight[];
};

layout(set = 0, binding = 3) buffer readonly biasBuffer {
    float bias[];
};

layout(push_constant) uniform args {
    uint B;
    uint T;
    uint C;
    uint OC;
};

void main() {
    const uint BT = B * T;
    const uint bt = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    const uint oc = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;

    if (bt < BT && oc < OC) {
        const uint b = bt / BT;
        const uint t = bt % BT;
        const uint weight_oc_offset = oc * C;
        const uint inp_bt_offset = b * BT * C + t * C;
        float val = bias[oc];
        for (uint c = 0; c < C; c++) {
            val += inp[inp_bt_offset + c] * weight[weight_oc_offset + c];
        }
        outs[bt * OC + oc] = val;
    }
}
