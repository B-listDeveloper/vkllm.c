#version 450
#pragma shader_stage(compute)

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

layout(set = 0, binding = 2) buffer writeonly b2 {
    float dbias[];
};

layout(set = 0, binding = 3) buffer readonly b3 {
    float dout[];
};

layout(push_constant) uniform args {
    uint B;
    uint T;
    uint C;
    uint OC;
};

void main() {
    const uint o = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    if (o < OC) {
        float sum = 0.0;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                sum += dout[b * T * OC + t * OC + o];
            }
        }
        dbias[o] = sum;
    }
}
