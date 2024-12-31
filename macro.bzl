load("@rules_cc//cc:defs.bzl", "cc_binary")
load(":shader.bzl", "glsl_shader")

def test_shader(name):
    glsl_shader(
        name = "{}_shaders".format(name),
        srcs = native.glob(["shaders/{}*.glsl".format(name)]),
    )

    cc_binary(
        name = name,
        srcs = [
            "vulkan/common.h",
            "vulkan/{}.c".format(name),
        ],
        copts = [
            "-O3",
            "-fopenmp",
        ],
        linkopts = [
            "-lgomp",
            "-lm",
            "-lvulkan",
        ],
        data = [
            ":{}_shaders".format(name),
        ],
    )
