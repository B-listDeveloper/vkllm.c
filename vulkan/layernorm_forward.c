/*
version 1 is naive port from CPU code to shader: parallelizes over B, T, loops over C
bazel run :layernorm_forward -- 1
*/
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 layernorm forward pass
void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                           float* inp, float* weight, float* bias,
                           int B, int T, int C) {
    // out: (B, T, C), mean: (B, T), rstd: (B, T), inp: (B, T, C), weight: (C), bias: (C)
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b, t, :]
            float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int c = 0; c < C; c++) {
                m += x[c];
            }
            m = m / C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int c = 0; c < C; c++) {
                float xshift = x[c] - m;
                v += xshift * xshift;
            }
            v = v / C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b, t, :]
            float* out_bt = out + b * T * C + t * C;
            for (int c = 0; c < C; c++) {
                float n = ((x[c] - m) * s);  // normalized output
                float o = n * weight[c] + bias[c];  // scale and shift it
                out_bt[c] = o;  // write
            }  // (1, 1, C)
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }  // (1, T, C)
    }  // (B, T, C)
}

// ----------------------------------------------------------------------------
// random utils

float* make_random_float(int N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        arr[i] = ((float)rand() / (float)RAND_MAX) * 2.0 - 1.0;  // [-1, 1]
    }
    return arr;
}

// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
    srand(0);

    // read shader_num from command line
    int shader_num = 1;
    if (argc > 1) {
        shader_num = atoi(argv[1]);
    }

    if (shader_num > 1) {
        printf("Invalid shader number\n");
        exit(EXIT_FAILURE);
    }
    printf("Using shader %d\n", shader_num);

    char filename[50];
    sprintf(filename, "shaders/layernorm_forward_shader%d.spv", shader_num);

    VkApplicationInfo appInfo = {0};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Compute Shader Matrix Multiplication";
    appInfo.applicationVersion = 0;
    appInfo.pEngineName = NULL;
    appInfo.engineVersion = 0;
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo instanceCreateInfo = {0};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pApplicationInfo = &appInfo;

    VkInstance instance;
    if (vkCreateInstance(&instanceCreateInfo, NULL, &instance) != VK_SUCCESS) {
        printf("Failed to create an instance!\n");
        exit(EXIT_FAILURE);
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
    if (deviceCount == 0) {
        printf("Failed to find GPUs with Vulkan support!\n");
        exit(EXIT_FAILURE);
    }

    VkPhysicalDevice pDevices[deviceCount];
    vkEnumeratePhysicalDevices(instance, &deviceCount, pDevices);

    uint32_t queueFamilyIdx;
    VkPhysicalDevice pDevice = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < deviceCount; i++) {
        VkPhysicalDevice* device = &pDevices[i];

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(*device, &queueFamilyCount, NULL);
        VkQueueFamilyProperties queueFamilies[queueFamilyCount];
        vkGetPhysicalDeviceQueueFamilyProperties(*device, &queueFamilyCount, queueFamilies);

        for (queueFamilyIdx = 0; queueFamilyIdx < queueFamilyCount; queueFamilyIdx++) {
            if (queueFamilies[queueFamilyIdx].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                break;
            }
        }

        if (queueFamilyIdx < queueFamilyCount) {
            pDevice = pDevices[i];
            break;
        }
    }

    if (pDevice == VK_NULL_HANDLE) {
        printf("Failed to find a suitable GPU!\n");
        exit(EXIT_FAILURE);
    }

    VkPhysicalDeviceProperties pDeviceProperties;
    vkGetPhysicalDeviceProperties(pDevice, &pDeviceProperties);
    printf("Device Name: %s\n", pDeviceProperties.deviceName);
    const uint32_t apiVersion = pDeviceProperties.apiVersion;
    printf("Vulkan Version: %d.%d.%d\n", VK_API_VERSION_MAJOR(apiVersion), VK_API_VERSION_MINOR(apiVersion), VK_API_VERSION_PATCH(apiVersion));
    const uint32_t shmSize = pDeviceProperties.limits.maxComputeSharedMemorySize / 1024;
    printf("Max Compute Shared Memory Size: %d KB\n", shmSize);
    printf("Compute Queue Family Index: %d\n", queueFamilyIdx);

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {0};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queueFamilyIdx;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &priority;

    VkDeviceCreateInfo deviceCreateInfo = {0};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;

    VkDevice device;
    if (vkCreateDevice(pDevice, &deviceCreateInfo, NULL, &device) != VK_SUCCESS) {
        printf("Failed to create logical device!\n");
        exit(EXIT_FAILURE);
    }

    // memory allocation
    const int32_t B = 8;
    const int32_t T = 1024;
    const int32_t C = 768;

    const uint32_t outSize = B * T * C * sizeof(float);
    const uint32_t meanSize = B * T * sizeof(float);
    const uint32_t rstdSize = B * T * sizeof(float);
    const uint32_t inpSize = B * T * C * sizeof(float);
    const uint32_t weightSize = C * sizeof(float);
    const uint32_t biasSize = C * sizeof(float);

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* mean = (float*)malloc(B * T * sizeof(float));
    float* rstd = (float*)malloc(B * T * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(C);
    float* bias = make_random_float(C);

    uint32_t tensorSizes[] = { outSize, meanSize, rstdSize, inpSize, weightSize, biasSize };
    uint32_t numTensors = sizeof(tensorSizes) / sizeof(uint32_t);
    uint32_t offsets[numTensors];
    uint32_t totalBufferSize = 0;
    for (int i = 0; i < numTensors; i++) {
        offsets[i] = totalBufferSize;
        totalBufferSize += tensorSizes[i];
    }

    VkBufferCreateInfo bufferCreateInfo = {0};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCreateInfo.size = totalBufferSize;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferCreateInfo.queueFamilyIndexCount = 1;
    bufferCreateInfo.pQueueFamilyIndices = &queueFamilyIdx;

    VkBuffer buffer;
    if (vkCreateBuffer(device, &bufferCreateInfo, NULL, &buffer) != VK_SUCCESS) {
        printf("Failed to create buffer!\n");
        exit(EXIT_FAILURE);
    }

    VkMemoryRequirements bufferMemReqs;
    vkGetBufferMemoryRequirements(device, buffer, &bufferMemReqs);

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(pDevice, &memProperties);

    uint32_t memoryTypeIndex = ~0;
    VkDeviceSize heapSize = ~0;
    for (int i = 0; i < memProperties.memoryTypeCount; i++) {
        VkMemoryType* t = &memProperties.memoryTypes[i];
        if ((VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT & t->propertyFlags) &&
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & t->propertyFlags) &&
                (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT & t->propertyFlags)) {
            heapSize = memProperties.memoryHeaps[t->heapIndex].size;
            memoryTypeIndex = i;
            break;
        }
    }
    printf("Memory Type Index: %d\n", memoryTypeIndex);
    printf("Memory Heap Size: %lu GB\n", heapSize / 1024 / 1024 / 1024);

    VkMemoryAllocateInfo mallocInfo = {0};
    mallocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mallocInfo.allocationSize = bufferMemReqs.size;
    mallocInfo.memoryTypeIndex = memoryTypeIndex;

    VkDeviceMemory deviceMem;
    if (vkAllocateMemory(device, &mallocInfo, NULL, &deviceMem) != VK_SUCCESS) {
        printf("Failed to allocate memory!\n");
        exit(EXIT_FAILURE);
    }

    vkBindBufferMemory(device, buffer, deviceMem, 0);

    void* d_out;
    void* d_mean;
    void* d_rstd;
    void* d_inp;
    void* d_weight;
    void* d_bias;
    vkMapMemory(device, deviceMem, offsets[0], tensorSizes[0], 0, &d_out);
    vkMapMemory(device, deviceMem, offsets[1], tensorSizes[1], 0, &d_mean);
    vkMapMemory(device, deviceMem, offsets[2], tensorSizes[2], 0, &d_rstd);
    vkMapMemory(device, deviceMem, offsets[3], tensorSizes[3], 0, &d_inp);
    vkMapMemory(device, deviceMem, offsets[4], tensorSizes[4], 0, &d_weight);
    vkMapMemory(device, deviceMem, offsets[5], tensorSizes[5], 0, &d_bias);

    memcpy((float*)d_mean, mean, meanSize);
    memcpy((float*)d_rstd, rstd, rstdSize);
    memcpy((float*)d_inp, inp, inpSize);
    memcpy((float*)d_weight, weight, weightSize);
    memcpy((float*)d_bias, bias, biasSize);

    FILE* shaderFile = fopen(filename, "rb");
    fseek(shaderFile, 0, SEEK_END);
    uint32_t shaderFileSize = ftell(shaderFile);
    fseek(shaderFile, 0, SEEK_SET);
    char shaderContents[shaderFileSize];
    fread(shaderContents, sizeof(char), shaderFileSize, shaderFile);
    fclose(shaderFile);

    VkShaderModuleCreateInfo shaderModuleCreateInfo = {0};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.pCode = (uint32_t*)shaderContents;
    shaderModuleCreateInfo.codeSize = shaderFileSize;

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &shaderModuleCreateInfo, NULL, &shaderModule) != VK_SUCCESS) {
        printf("Failed to create shader module!\n");
        exit(EXIT_FAILURE);
    }

    VkDescriptorSetLayoutBinding layoutBindings[numTensors];
    for (uint32_t i = 0; i < numTensors; i++) {
        layoutBindings[i] = (VkDescriptorSetLayoutBinding){
            i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL
        };
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo = {0};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = numTensors;
    layoutInfo.pBindings = layoutBindings;

    VkDescriptorSetLayout computeDescriptorSetLayout;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, NULL, &computeDescriptorSetLayout) != VK_SUCCESS) {
        printf("Failed to create compute descriptor set layout!\n");
        exit(EXIT_FAILURE);
    }

    struct {
        uint32_t N;
        uint32_t C;
    } shapes = { B * T, C };

    VkPushConstantRange pushConstantRange = {0};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(shapes);

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {0};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &computeDescriptorSetLayout;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

    VkPipelineLayout pipelineLayout;
    vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &pipelineLayout);

    // necessary?
    VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {0};
    pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;

    VkPipelineCache pipelineCache;
    vkCreatePipelineCache(device, &pipelineCacheCreateInfo, NULL, &pipelineCache);

    struct {
        uint32_t workgroup_x;
    } wgSizes = { 256 };

    VkSpecializationMapEntry entries[] = {
        { 1, 0, 4 },
    };

    VkSpecializationInfo specializationInfo = {0};
    specializationInfo.mapEntryCount = 1;
    specializationInfo.pMapEntries = entries;
    specializationInfo.dataSize = sizeof(wgSizes);
    specializationInfo.pData = &wgSizes;

    VkComputePipelineCreateInfo computePipelineCreateInfo = {0};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computePipelineCreateInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computePipelineCreateInfo.stage.module = shaderModule;
    computePipelineCreateInfo.stage.pName = "main";
    computePipelineCreateInfo.stage.pSpecializationInfo = &specializationInfo;
    computePipelineCreateInfo.layout = pipelineLayout;

    VkPipeline computePipeline;
    vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, NULL, &computePipeline);

    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, numTensors },
    };

    VkDescriptorPoolCreateInfo poolInfo = {0};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;  // how many sets to be allocated
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;

    VkDescriptorPool descriptorPool;
    if (vkCreateDescriptorPool(device, &poolInfo, NULL, &descriptorPool) != VK_SUCCESS) {
        printf("Failed to create descriptor pool!\n");
        exit(EXIT_FAILURE);
    }

    VkDescriptorSetAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &computeDescriptorSetLayout;

    VkDescriptorSet computeDescriptorSet;
    if (vkAllocateDescriptorSets(device, &allocInfo, &computeDescriptorSet) != VK_SUCCESS) {
        printf("Failed to allocate descriptor sets!\n");
        exit(EXIT_FAILURE);
    }

    VkDescriptorBufferInfo bufferInfo[numTensors];
    VkWriteDescriptorSet writeDescSets[numTensors];
    for (uint32_t i = 0; i < numTensors; i++) {
        bufferInfo[i] = (VkDescriptorBufferInfo){ buffer, offsets[i], tensorSizes[i] };
        writeDescSets[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, computeDescriptorSet,
            i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bufferInfo[i], NULL
        };
    }
    vkUpdateDescriptorSets(device, numTensors, writeDescSets, 0, NULL);

    VkCommandPoolCreateInfo commandPoolCreateInfo = {0};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.queueFamilyIndex = queueFamilyIdx;

    VkCommandPool commandPool;
    vkCreateCommandPool(device, &commandPoolCreateInfo, NULL, &commandPool);

    VkCommandBufferAllocateInfo commandBufferAllocInfo = {0};
    commandBufferAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocInfo.commandPool = commandPool;
    commandBufferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &commandBufferAllocInfo, &commandBuffer);

    VkCommandBufferBeginInfo commandBufferBeginInfo = {0};
    commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    VkQueryPoolCreateInfo queryPoolCreateInfo = {0};
    queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolCreateInfo.queryCount = 2;  // start and stop

    VkQueryPool queryPool;
    if (vkCreateQueryPool(device, &queryPoolCreateInfo, NULL, &queryPool) != VK_SUCCESS) {
        printf("Failed to create query pool!\n");
        exit(EXIT_FAILURE);
    }

    vkResetCommandBuffer(commandBuffer, 0);

    vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
    vkCmdResetQueryPool(commandBuffer, queryPool, 0, 2);
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &computeDescriptorSet, 0, NULL);
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shapes), &shapes);
    vkCmdDispatch(commandBuffer, CEIL_DIV(B * T, 256), 1, 1);

    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo = {0};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VkQueue queue;
    vkGetDeviceQueue(device, queueFamilyIdx, 0, &queue);

    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);
    float* out_gpu = (float*)malloc(outSize);
    memcpy(out_gpu, d_out, outSize);
    for (int i = 0; i < B * T * C; i++) {
        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", out[i], out_gpu[i]);
        }
        // ensure correctness for all elements
        if (i >= 5 && fabsf(out[i] - out_gpu[i]) > 1e-4) {
            printf("Mismatch at %d: %f vs %f\n", i, out[i], out_gpu[i]);
            exit(1);
        }
    }
    printf("Results match at block_size=256!\n");

    // time the shader at different block sizes
    int block_sizes[] = { 32, 64, 128, 256, 512, 1024 };
    int repeat_times = 10;

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        wgSizes.workgroup_x = block_size;

        // Is no destruction needed before re-creation?
        vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, NULL, &computePipeline);

        vkResetCommandBuffer(commandBuffer, 0);

        vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
        vkCmdResetQueryPool(commandBuffer, queryPool, 0, 2);
        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &computeDescriptorSet, 0, NULL);
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shapes), &shapes);
        vkCmdDispatch(commandBuffer, CEIL_DIV(B * T, block_size), 1, 1);

        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
        vkEndCommandBuffer(commandBuffer);

        uint64_t elapsed_ns = 0;
        uint64_t times[2];
        for (int i = 0; i < repeat_times; i++) {
            vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(queue);
            vkGetQueryPoolResults(device, queryPool, 0, 2, 2 * sizeof(uint64_t), times, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
            elapsed_ns += times[1] - times[0];
        }
        float elapsed_time = (float)elapsed_ns / 1e6;

        // double-check if the result is correct
        memcpy(out_gpu, d_out, outSize);
        for (int i = 0; i < B * T * C; i++) {
            if (fabsf(out[i] - out_gpu[i]) > 1e-4) {
                printf("Mismatch at %d: %f vs %f\n", i, out[i], out_gpu[i]);
                exit(EXIT_FAILURE);
            }
        }

        // napkin math: estimate the memory bandwidth achieved
        // e.g. A100 40GB PCIe is advertised at 1,555 GB/s
        long memory_ops = (2 * B * T * C) * 4;  // * 4 for float
        float memory_bandwidth = memory_ops / (elapsed_time / repeat_times) / 1e6;

        printf("block_size %4d | time %f ms | bandwidth %f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    vkUnmapMemory(device, deviceMem);
    d_out = d_mean = d_rstd = d_inp = d_weight = d_bias = NULL;

    vkDestroyQueryPool(device, queryPool, NULL);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    vkDestroyCommandPool(device, commandPool, NULL);
    vkDestroyPipeline(device, computePipeline, NULL);
    vkDestroyPipelineCache(device, pipelineCache, NULL);
    vkDestroyPipelineLayout(device, pipelineLayout, NULL);
    vkDestroyShaderModule(device, shaderModule, NULL);
    vkFreeDescriptorSets(device, descriptorPool, 1, &computeDescriptorSet);
    vkDestroyDescriptorPool(device, descriptorPool, NULL);
    vkDestroyDescriptorSetLayout(device, computeDescriptorSetLayout, NULL);
    vkFreeMemory(device, deviceMem, NULL);
    vkDestroyBuffer(device, buffer, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);

    free(out);
    free(d_mean);
    free(d_rstd);
    free(inp);
    free(weight);
    free(bias);
    free(out_gpu);

    return 0;
}
