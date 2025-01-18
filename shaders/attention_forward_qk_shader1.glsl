#version 450
#pragma shader_stage(compute)

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

layout(set = 0, binding = 1) buffer writeonly b1 {
    float preatt[];
};

layout(set = 0, binding = 3) buffer readonly b3 {
    float inp[];
};

layout(push_constant) uniform args {
    uint B;
    uint T;
    uint C;
    uint H;
};

void main() {
    const uint idx = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    if (idx < B * H * T * T) {
        const uint t2 = idx % T;
        const uint t = (idx / T) % T;
        if (t2 > t) return;  // autoregressive mask
        const uint h = (idx / (T * T)) % H;
        const uint b = idx / (H * T * T);

        const uint C3 = 3 * C;
        const uint hs = C / H;  // head size
        const uint query_t_offset = b * T * C3 + t * C3 + h * hs;
        const uint key_t2_offset = b * T * C3 + t2 * C3 + h * hs + C;  // + C because it's key

        // (query_t) dot (key_t2)
        float val = 0.0f;
        for (int c = 0; c < hs; c++) {
            val += inp[query_t_offset + c] * inp[key_t2_offset + c];
        }
        val *= 1.0 / sqrt(hs);

        preatt[idx] = val;
    }
}
