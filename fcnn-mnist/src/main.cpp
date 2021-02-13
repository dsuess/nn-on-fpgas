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
#include <tuple>
#include <nonstd/optional.hpp>

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

std::pair<Matrix, cl::Event> apply_matmul(Matrix &matrixA, Matrix &matrixB, DeviceHandle &handle, cl::Kernel &kernel, std::vector<cl::Event> *wait_on = NULL)
{
    Matrix result = Matrix(matrixA.rows, matrixB.cols, 4096);
    result.to_device(handle);
    kernel.setArg(0, matrixA.get_buffer());
    kernel.setArg(1, matrixB.get_buffer());
    kernel.setArg(2, matrixA.rows);
    kernel.setArg(3, matrixA.cols);
    kernel.setArg(4, matrixB.cols);
    kernel.setArg(5, result.get_buffer());

    cl::Event event;
    handle.q.enqueueTask(kernel, wait_on, &event);
    return std::make_pair(std::move(result), std::move(event));
}

cl::Event apply_bias(Matrix &input, Matrix &bias, DeviceHandle &handle, cl::Kernel &kernel, std::vector<cl::Event> *wait_on = NULL)
{
    kernel.setArg(0, input.get_buffer());
    kernel.setArg(1, bias.get_buffer());
    kernel.setArg(2, input.rows);
    kernel.setArg(3, input.cols);

    cl::Event event;
    handle.q.enqueueTask(kernel, wait_on, &event);
    return std::move(event);
}

int main(int argc, const char *argv[])
{
    DeviceHandle handle = setup_handle();
    const std::string devName = handle.device.getInfo<CL_DEVICE_NAME>();
    auto xclBins = xcl::import_binary_file("xclbin/kernels.xclbin");
    cl::Program program(handle.context, {handle.device}, xclBins);
    cl::Kernel matmul_kernel(program, "matmul_kernel");
    cl::Kernel bias_relu6_kernel(program, "bias_relu6_kernel");
    cl::Kernel bias_softmax_kernel(program, "bias_softmax_kernel");

    const uint batch_size = 4;
    Matrix inputs = Matrix::constant(batch_size, 4, 1.);
    Matrix weights = Matrix::constant(4, 2, 10);
    Matrix biases = Matrix(2, 1);
    biases.set_value(0, 0, 1.);
    biases.set_value(1, 0, 2.);
    inputs.to_device(handle);
    weights.to_device(handle);
    biases.to_device(handle);
    handle.q.finish();

    Matrix result;
    std::vector<cl::Event> events;
    {
        events.resize(1);
        std::tie(result, events[0]) = apply_matmul(inputs, weights, handle, matmul_kernel);
        apply_bias(result, biases, handle, bias_relu6_kernel, &events);
        handle.q.finish();
        result.to_cpu(handle);
        handle.q.finish();
        std::cout << "softmax:\n~~~~~~~\n"
                  << result.to_string() << std::endl;
    }

    // Data transfer from device buffer to host buffer

    std::cout << "DONE" << std::endl;
}
