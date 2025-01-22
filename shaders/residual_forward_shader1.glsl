#version 450
#pragma shader_stage(compute)

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

layout(set = 0, binding = 0) buffer writeonly b0 {
    float outs[];
};

layout(set = 0, binding = 1) buffer readonly b1 {
    float inp1[];
};

layout(set = 0, binding = 2) buffer readonly b2 {
    float inp2[];
};

layout(push_constant) uniform args {
    uint N;
};

void main() {
    const uint idx = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    if (idx < N) {
        outs[idx] = inp1[idx] + inp2[idx];
    }
}
