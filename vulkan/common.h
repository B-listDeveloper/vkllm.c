#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

typedef struct Group {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} Group;

typedef struct Launcher {
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkCommandBufferBeginInfo begin_info;
    VkSubmitInfo submit_info;
    VkQueue queue;
#ifdef TIMER
    VkQueryPool query_pool;
#endif
    bool recorded;
} Launcher;

typedef struct Kernel {
    VkPipelineLayout layout;
    VkPipeline* pipelines;
    uint32_t reserved;
    uint32_t count;
} Kernel;

typedef struct Memory {
    VkBuffer buffer;
    VkDeviceMemory heap;
    VkDeviceSize size;

    VkDescriptorPool pool;
    VkDescriptorSetLayout layout;
    VkDescriptorSet set;
} Memory;

typedef struct Device {
    VkPhysicalDevice physical_device;
    VkDevice logical_device;

    uint32_t queue_family_index;
    uint32_t memory_type_index;
} Device;

typedef struct Context {
    VkInstance instance;
    Device device;
} Context;

void init_instance(Context* context) {
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

    // VkInstance instance;
    if (vkCreateInstance(&instanceCreateInfo, NULL, &context->instance) != VK_SUCCESS) {
        printf("Failed to create an instance!\n");
        exit(EXIT_FAILURE);
    }
}

void init_device(Context* context) {
    context->device.physical_device = VK_NULL_HANDLE;

    uint32_t num_devices = 0;
    vkEnumeratePhysicalDevices(context->instance, &num_devices, NULL);
    if (num_devices == 0) {
        printf("Failed to find GPUs with Vulkan support!\n");
        exit(EXIT_FAILURE);
    }
    VkPhysicalDevice physical_devices[num_devices];
    vkEnumeratePhysicalDevices(context->instance, &num_devices, physical_devices);

    VkDeviceSize heap_size = ~0;
    for (uint32_t i = 0; i < num_devices; i++) {
        context->device.memory_type_index = context->device.queue_family_index = -1;

        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(physical_devices[i], &mem_props);

        for (uint32_t j = 0; j < mem_props.memoryTypeCount; j++) {
            if ((mem_props.memoryTypes[j].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                    (mem_props.memoryTypes[j].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                    (mem_props.memoryTypes[j].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                context->device.memory_type_index = j;
                break;
            }
        }

        uint32_t num_queue_families = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[i], &num_queue_families, NULL);
        VkQueueFamilyProperties queue_families[num_queue_families];
        vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[i], &num_queue_families, queue_families);

        for (uint32_t j = 0; j < num_queue_families; j++) {
            if (queue_families[j].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                context->device.queue_family_index = j;
                heap_size = mem_props.memoryHeaps[j].size;
                break;
            }
        }

        if (context->device.memory_type_index >= 0 &&
                context->device.queue_family_index >= 0) {
            context->device.physical_device = physical_devices[i];
            break;
        }
    }

    if (context->device.physical_device == VK_NULL_HANDLE) {
        printf("Failed to find a suitable GPU!\n");
        exit(EXIT_FAILURE);
    }

    VkPhysicalDeviceProperties physical_device_props;
    vkGetPhysicalDeviceProperties(context->device.physical_device, &physical_device_props);
    printf("Device Name: %s\n", physical_device_props.deviceName);
    const uint32_t apiVersion = physical_device_props.apiVersion;
    printf("Vulkan Version: %d.%d.%d\n", VK_API_VERSION_MAJOR(apiVersion), VK_API_VERSION_MINOR(apiVersion), VK_API_VERSION_PATCH(apiVersion));
    const uint32_t shmSize = physical_device_props.limits.maxComputeSharedMemorySize / 1024;
    printf("Max Compute Shared Memory Size: %d KB\n", shmSize);
    printf("Compute Queue Family Index: %d\n", context->device.queue_family_index);
    printf("Memory Type Index: %d\n", context->device.memory_type_index);
    printf("Memory Heap Size: %lu GB\n", heap_size / 1024 / 1024 / 1024);

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {0};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = context->device.queue_family_index;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &priority;

    VkDeviceCreateInfo deviceCreateInfo = {0};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;

    if (vkCreateDevice(context->device.physical_device, &deviceCreateInfo, NULL, &context->device.logical_device) != VK_SUCCESS) {
        printf("Failed to create logical device!\n");
        exit(EXIT_FAILURE);
    }
}

void init_context(Context* context) {
    init_instance(context);
    init_device(context);
}

void destroy_context(Context* context) {
    vkDestroyDevice(context->device.logical_device, NULL);
    vkDestroyInstance(context->instance, NULL);
}

void allocate_memory(Context* context, Memory *memory, uint32_t num_tensors, uint32_t* sizes) {
    uint32_t offsets[num_tensors];
    VkDeviceSize total_size = 0;
    for (int i = 0; i < num_tensors; i++) {
        offsets[i] = total_size;
        total_size += sizes[i];
    }
    memory->size = total_size;

    VkBufferCreateInfo buffer_create_info = {0};
    buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_create_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_create_info.size = memory->size;
    buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    buffer_create_info.queueFamilyIndexCount = 1;
    buffer_create_info.pQueueFamilyIndices = &context->device.queue_family_index;

    if (vkCreateBuffer(context->device.logical_device, &buffer_create_info, NULL, &memory->buffer) != VK_SUCCESS) {
        printf("Failed to create buffer!\n");
        exit(EXIT_FAILURE);
    }

    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(context->device.logical_device, memory->buffer, &requirements);

    VkMemoryAllocateInfo malloc_info = {0};
    malloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    malloc_info.allocationSize = requirements.size;
    malloc_info.memoryTypeIndex = context->device.memory_type_index;

    if (vkAllocateMemory(context->device.logical_device, &malloc_info, NULL, &memory->heap) != VK_SUCCESS) {
        printf("Failed to allocate memory!\n");
        exit(EXIT_FAILURE);
    }

    vkBindBufferMemory(context->device.logical_device, memory->buffer, memory->heap, 0);

    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, num_tensors },
    };

    VkDescriptorPoolCreateInfo pool_info = {0};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;  // how many sets to be allocated
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = pool_sizes;

    if (vkCreateDescriptorPool(context->device.logical_device, &pool_info, NULL, &memory->pool) != VK_SUCCESS) {
        printf("Failed to create descriptor pool!\n");
        exit(EXIT_FAILURE);
    }

    VkDescriptorSetLayoutBinding bindings[num_tensors];
    for (uint32_t i = 0; i < num_tensors; i++) {
        bindings[i] = (VkDescriptorSetLayoutBinding){
            i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL
        };
    }

    VkDescriptorSetLayoutCreateInfo layout_info = {0};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = num_tensors;
    layout_info.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(context->device.logical_device, &layout_info, NULL, &memory->layout) != VK_SUCCESS) {
        printf("Failed to create compute descriptor set layout!\n");
        exit(EXIT_FAILURE);
    }

    VkDescriptorSetAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = memory->pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &memory->layout;

    if (vkAllocateDescriptorSets(context->device.logical_device, &alloc_info, &memory->set) != VK_SUCCESS) {
        printf("Failed to allocate descriptor sets!\n");
        exit(EXIT_FAILURE);
    }

    VkDescriptorBufferInfo buffer_info[num_tensors];
    VkWriteDescriptorSet writes[num_tensors];
    for (uint32_t i = 0; i < num_tensors; i++) {
        buffer_info[i] = (VkDescriptorBufferInfo){ memory->buffer, offsets[i], sizes[i] };
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, memory->set,
            i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &buffer_info[i], NULL
        };
    }
    vkUpdateDescriptorSets(context->device.logical_device, num_tensors, writes, 0, NULL);
}

void free_memory(Context* context, Memory* memory) {
    vkFreeDescriptorSets(context->device.logical_device, memory->pool, 1, &memory->set);
    vkDestroyDescriptorPool(context->device.logical_device, memory->pool, NULL);
    vkDestroyDescriptorSetLayout(context->device.logical_device, memory->layout, NULL);
    vkFreeMemory(context->device.logical_device, memory->heap, NULL);
    vkDestroyBuffer(context->device.logical_device, memory->buffer, NULL);
}

void init_kernel(Context* context, Memory* memory, Kernel* kernel) {
    kernel->count = 0;
    kernel->reserved = 1;
    kernel->pipelines = malloc(sizeof(VkPipeline));

    VkPushConstantRange range = {0};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.offset = 0;
    range.size = 16 * sizeof(uint32_t);

    VkPipelineLayoutCreateInfo pipeline_layout_info = {0};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &memory->layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &range;

    vkCreatePipelineLayout(context->device.logical_device, &pipeline_layout_info, NULL, &kernel->layout);
}

void append_shader(Context* context, Kernel* kernel, const char* filename, Group wg_sizes) {
    if (kernel->reserved == kernel->count) {
        kernel->reserved *= 2;
        kernel->pipelines = realloc(
            kernel->pipelines, kernel->reserved * sizeof(VkPipeline));
    }

    FILE* shader_file = fopen(filename, "rb");

    fseek(shader_file, 0, SEEK_END);
    uint32_t file_size = ftell(shader_file);

    char contents[file_size];
    fseek(shader_file, 0, SEEK_SET);
    fread(contents, sizeof(char), file_size, shader_file);
    fclose(shader_file);

    VkShaderModuleCreateInfo shader_info = {0};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.pCode = (uint32_t*)contents;
    shader_info.codeSize = file_size;

    VkShaderModule shader;
    if (vkCreateShaderModule(context->device.logical_device, &shader_info, NULL, &shader) != VK_SUCCESS) {
        printf("Failed to create shader module!\n");
        exit(EXIT_FAILURE);
    }

    VkSpecializationMapEntry entries[] = { { 1, 0, 4 }, { 2, 4, 4 }, { 3, 8, 4 } };
    VkSpecializationInfo workgroup_info = {0};
    workgroup_info.mapEntryCount = 3;
    workgroup_info.pMapEntries = entries;
    workgroup_info.dataSize = sizeof(Group);
    workgroup_info.pData = &wg_sizes;

    VkComputePipelineCreateInfo pipeline_info = {0};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = shader;
    pipeline_info.stage.pName = "main";
    pipeline_info.stage.pSpecializationInfo = &workgroup_info;
    pipeline_info.layout = kernel->layout;

    vkCreateComputePipelines(context->device.logical_device, VK_NULL_HANDLE, 1,
                             &pipeline_info, NULL, &kernel->pipelines[kernel->count++]);

    // https://github.com/blurrypiano/littleVulkanEngine/issues/29
    vkDestroyShaderModule(context->device.logical_device, shader, NULL);
}

void destroy_kernel(Context* context, Kernel* kernel) {
    for (int i = 0; i < kernel->count; i++) {
        vkDestroyPipeline(context->device.logical_device, kernel->pipelines[i], NULL);
    }
    vkDestroyPipelineLayout(context->device.logical_device, kernel->layout, NULL);

    free(kernel->pipelines);
}

void init_launcher(Context* context, Launcher* launcher) {
    launcher->recorded = false;

    VkCommandPoolCreateInfo pool_info = {0};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = context->device.queue_family_index;

    vkCreateCommandPool(context->device.logical_device, &pool_info, NULL, &launcher->command_pool);

    VkCommandBufferAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = launcher->command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    vkAllocateCommandBuffers(context->device.logical_device, &alloc_info, &launcher->command_buffer);

    memset(&launcher->begin_info, 0, sizeof(VkCommandBufferBeginInfo));
    launcher->begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    launcher->begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    memset(&launcher->submit_info, 0, sizeof(VkSubmitInfo));
    launcher->submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    launcher->submit_info.commandBufferCount = 1;
    launcher->submit_info.pCommandBuffers = &launcher->command_buffer;

    vkGetDeviceQueue(context->device.logical_device, context->device.queue_family_index, 0, &launcher->queue);

#ifdef TIMER
    VkQueryPoolCreateInfo query_pool_info = {0};
    query_pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query_pool_info.queryCount = 2;  // start and stop

    if (vkCreateQueryPool(context->device.logical_device, &query_pool_info, NULL, &launcher->query_pool) != VK_SUCCESS) {
        printf("Failed to create query pool!\n");
        exit(EXIT_FAILURE);
    }
#endif
}

void launch_kernel(Context* context, Memory* memory,
                   Kernel* kernel, Launcher* launcher,
                   uint32_t num_constants, uint32_t* constants,
                   Group group_counts,
                   bool force_reset) {
    uint32_t action = force_reset ? 0 : (launcher->recorded ? 2 : 1);
    uint32_t constant_size = num_constants * sizeof(uint32_t);

    switch (action) {
    case 0:
        vkResetCommandBuffer(launcher->command_buffer, 0);
    case 1:
        vkBeginCommandBuffer(launcher->command_buffer, &launcher->begin_info);
#ifdef TIMER
        vkCmdResetQueryPool(launcher->command_buffer, launcher->query_pool, 0, 2);
        vkCmdWriteTimestamp(launcher->command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, launcher->query_pool, 0);
#endif
        vkCmdBindDescriptorSets(launcher->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->layout, 0, 1, &memory->set, 0, NULL);
        vkCmdPushConstants(launcher->command_buffer, kernel->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, constant_size, constants);
        for (int i = 0; i < kernel->count; i++) {
            vkCmdBindPipeline(launcher->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->pipelines[i]);
            vkCmdDispatch(launcher->command_buffer, group_counts.x, group_counts.y, group_counts.z);
            vkCmdPipelineBarrier(
                launcher->command_buffer, VK_SHADER_STAGE_COMPUTE_BIT, VK_SHADER_STAGE_COMPUTE_BIT,
                0, 0, NULL, 0, NULL, 0, NULL
            );
        }
#ifdef TIMER
        vkCmdWriteTimestamp(launcher->command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, launcher->query_pool, 1);
#endif
        vkEndCommandBuffer(launcher->command_buffer);

        launcher->recorded = true;
    default:
        vkQueueSubmit(launcher->queue, 1, &launcher->submit_info, VK_NULL_HANDLE);
        vkQueueWaitIdle(launcher->queue);
    }
}

void destroy_launcher(Context* context, Launcher* launcher) {
#ifdef TIMER
    vkDestroyQueryPool(context->device.logical_device, launcher->query_pool, NULL);
#endif
    vkFreeCommandBuffers(context->device.logical_device, launcher->command_pool, 1, &launcher->command_buffer);
    vkDestroyCommandPool(context->device.logical_device, launcher->command_pool, NULL);
}
