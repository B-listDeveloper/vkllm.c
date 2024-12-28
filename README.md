# vkllm.c

A reproduction of llm.c by @karpathy using Vulkan compute shaders.

## Prerequisite

Install `vulkan-sdk` following https://vulkan.lunarg.com/sdk/home#linux.

```bash
wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-{RELEASE}-{CODENAME}.list https://packages.lunarg.com/vulkan/{RELEASE}/lunarg-vulkan-{RELEASE}-{CODENAME}.list
sudo apt update
sudo apt install vulkan-sdk
```

And don't forget to install GPU driver.

## Usage

I only added `matmul_forward` for now, and you can test it by running:

```bash
bazel run :matmul_forward -- 1
```

Many more to come...

## Notes

If you see `llvmpipe` in stdout when running, your GPU is not being used correctly.
