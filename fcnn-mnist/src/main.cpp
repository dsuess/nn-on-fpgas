/*
 * Copyright 2020 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <string.h>
#include <sys/time.h>
#include <algorithm>

#include "xcl2.hpp"
#include "utils.hpp"

typedef struct DeviceHandle
{
    cl::Device device;
    cl::CommandQueue q;
    cl::Context context;
} CLContext;

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

cl::Kernel load_kernel(const std::string &name, const DeviceHandle &handle)
{
    const std::string devName = handle.device.getInfo<CL_DEVICE_NAME>();
    std::string xclbin_path = xcl::find_binary_file(devName, name);
    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    cl::Program program(handle.context, {handle.device}, xclBins);
    cl::Kernel kernel(program, name.c_str());
    std::cout << "INFO: Kernel '" << name << "' has been created" << std::endl;
    return kernel;
}

int main(int argc, const char *argv[])
{
    DeviceHandle handle = setup_handle();
    cl::Kernel kernel = load_kernel("matmul_kernel", handle);

    // Initialization of host buffers
    double *dataA;
    const int inout_size = 10;
    dataA = aligned_alloc<double>(inout_size);

    // DDR Settings
    std::vector<cl_mem_ext_ptr_t> mext_io(1);
    mext_io[0].flags = XCL_MEM_DDR_BANK1;
    mext_io[0].obj = dataA;
    mext_io[0].param = 0;

    // Create device buffer and map dev buf to host buf
    std::vector<cl::Buffer> buffer(1);
    buffer[0] = cl::Buffer(handle.context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                           sizeof(double) * inout_size, &mext_io[0]);

    // Data transfer from host buffer to device buffer
    std::vector<std::vector<cl::Event>> kernel_evt(2);
    kernel_evt[0].resize(1);
    kernel_evt[1].resize(1);

    std::vector<cl::Memory> ob_io;
    ob_io.push_back(buffer[0]);

    handle.q.enqueueMigrateMemObjects(ob_io, 0, nullptr, &kernel_evt[0][0]); // 0 : migrate from host to dev
    handle.q.finish();
    std::cout << "INFO: Finish data transfer from host to device" << std::endl;

    // Setup kernel
    // cholesky_kernel.setArg(0, dataAN);
    // cholesky_kernel.setArg(1, buffer[0]);
    // handle.q.finish();
    // std::cout << "INFO: Finish kernel setup" << std::endl;

    // handle.q.enqueueTask(kernel, nullptr, nullptr);
    handle.q.finish();

    // Data transfer from device buffer to host buffer
    handle.q.enqueueMigrateMemObjects(ob_io, 1, nullptr, nullptr); // 1 : migrate from dev to host
    handle.q.finish();

    std::cout << "DONE" << std::endl;
}
