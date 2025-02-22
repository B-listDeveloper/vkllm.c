#version 450
#pragma shader_stage(compute)

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

layout(set = 0, binding = 0) buffer writeonly b0 {
    float dinp1[];
};

layout(set = 0, binding = 1) buffer writeonly b1 {
    float dinp2[];
};

layout(set = 0, binding = 2) buffer readonly b2 {
    float dout[];
};

layout(push_constant) uniform args {
    uint N;
};

void main() {
    const uint idx = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    if (idx < N) {
        dinp1[idx] += dout[idx];
        dinp2[idx] += dout[idx];
    }
}
