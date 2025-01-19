#version 450
#pragma shader_stage(compute)

// would it be better to use radians(180)?
#define PI 3.14159265358979323846
#define GELU_SCALING_FACTOR sqrt(2.0f / PI)

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

layout(set = 0, binding = 0) buffer writeonly b0 {
    float outs[];
};

layout(set = 0, binding = 1) buffer readonly b1 {
    float inp[];
};

layout(push_constant) uniform args {
    uint N;
};

void main() {
    const uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    if (i < N) {
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        outs[i] = 0.5f * xi * (1.0f + tanh(GELU_SCALING_FACTOR * (xi + cube)));
    }
}
