#version 450
#pragma shader_stage(compute)

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

layout(set = 0, binding = 2) buffer writeonly b2 {
    float att[];
};

layout(set = 0, binding = 1) buffer readonly b1 {
    float preatt[];
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
        const uint t = idx % T;
        const uint h = (idx / T) % H;
        const uint b = idx / (H * T);

        const uint bht_offset = b * H * T * T + h * T * T + t * T;

        // find maxval
        float maxval = -10000.0f;  // TODO something better
        for (int t2 = 0; t2 <= t; t2++) {
            if (preatt[bht_offset + t2] > maxval) {
                maxval = preatt[bht_offset + t2];
            }
        }

        // calculate the exp and keep track of sum
        float expsum = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
            float expv = exp(preatt[bht_offset + t2] - maxval);
            expsum += expv;
            att[bht_offset + t2] = expv;
        }
        float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

        // normalize to get the softmax
        for (int t2 = 0; t2 < T; t2++) {
            if (t2 <= t) {
                att[bht_offset + t2] *= expsum_inv;
            } else {
                // causal attention mask. not strictly necessary to set to zero here
                // only doing this explicitly for debugging and checking to PyTorch
                att[bht_offset + t2] = 0.0f;
            }
        }
    }
}
