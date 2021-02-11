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
#include "matrix.hpp"

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

    Matrix matrixA = Matrix::constant(3, 3, 10., 4096);
    std::cout << matrixA.to_string();

    auto buffer = matrixA.get_cl_buffer(handle, XCL_MEM_DDR_BANK1);

    // Setup kernel
    // cholesky_kernel.setArg(0, dataAN);
    // cholesky_kernel.setArg(1, buffer[0]);
    // handle.q.finish();
    // std::cout << "INFO: Finish kernel setup" << std::endl;

    // handle.q.enqueueTask(kernel, nullptr, nullptr);
    handle.q.finish();

    // Data transfer from device buffer to host buffer

    std::cout << "DONE" << std::endl;
}
