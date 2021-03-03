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

#include "xcl2.hpp"
#include "matrix.hpp"
#include "net.hpp"

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

int main(int argc, const char *argv[])
{
    DeviceHandle handle = setup_handle();
    init_kernels(handle);

    auto model = FCNN("weights/");
    auto input = Matrix::from_npy("weights/samples.npy");
    input.to_device(handle);
    auto result = model(input);

    handle.q.finish();
    result.to_cpu(handle);
    handle.q.finish();

    // print argmax result
    for (int i = 0; i < result.rows; i++)
    {
        float minval = -1;
        int idx = -1;

        for (int j = 0; j < result.cols; j++)
        {
            auto val = result(i, j);
            if (minval < val)
            {
                idx = j;
                minval = val;
            }
        }

        std::cout << idx << " ";
    }
    std::cout << std::endl;
}
