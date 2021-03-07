#ifndef nn_on_fpga_utils
#define nn_on_fpga_utils

#include <iostream>
#include <tuple>
#include "xcl2.hpp"

typedef struct DeviceHandle
{
    cl::Device device;
    cl::CommandQueue q;
    cl::Context context;
} DeviceHandle;

static cl::Kernel MATMUL_KERNEL, BIAS_RELU6_KERNEL, BIAS_SOFTMAX_KERNEL;
static DeviceHandle HANDLE;

void init_kernels(DeviceHandle &handle)
{
    HANDLE = handle;
    auto xclBins = xcl::import_binary_file("xclbin/kernels.xclbin");
    cl::Program program(handle.context, {handle.device}, xclBins);
    MATMUL_KERNEL = cl::Kernel(program, "matmul_kernel");
    BIAS_RELU6_KERNEL = cl::Kernel(program, "bias_relu6_kernel");
    BIAS_SOFTMAX_KERNEL = cl::Kernel(program, "bias_softmax_kernel");
}

DeviceHandle setup_handle()
{
    DeviceHandle result;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    result.device = devices[0];

    // Creating Context and Command Queue for selected Device
    result.context = cl::Context(result.device);
    result.q = cl::CommandQueue(result.context, result.device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    std::string devName = result.device.getInfo<CL_DEVICE_NAME>();
    std::cout << "INFO: Found Device=" << devName << std::endl;
    return result;
}

#endif /* end of include guard: nn_on_fpga_utils */