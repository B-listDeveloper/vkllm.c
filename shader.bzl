def _impl(ctx):
    outs = []
    for src in ctx.files.srcs:
        out_name = src.basename[:-len(src.extension)]
        out = ctx.actions.declare_file(out_name + "spv", sibling=src)
        outs.append(out)

        ctx.actions.run(
            executable = ctx.attr.compiler,
            inputs = ctx.files.srcs,
            outputs = [out],
            arguments = [src.path, "-o", out.path],
        )

    return DefaultInfo(
        files = depset(outs),
        runfiles = ctx.runfiles(files=outs),
    )

glsl_shader = rule(
    implementation = _impl,
    attrs = {
        "compiler": attr.string(
            default = "glslc",
        ),
        "srcs": attr.label_list(
            allow_files = [".glsl"],
            mandatory = True,
        ),
    },
)
