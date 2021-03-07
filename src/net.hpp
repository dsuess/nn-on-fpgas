#ifndef NNONFPGA_NET
#define NNONFPGA_NET

#include <CL/cl2.hpp>
#include <vector>
#include "matrix.hpp"
#include "utils.hpp"
#include "xcl2.hpp"

class FCNN
{
private:
    Matrix weight1, weight2, bias1, bias2;

public:
    FCNN()
    {
        weight1 = Matrix::constant(784, 64, 1.0);
        weight1.to_device(HANDLE);
        bias1 = Matrix::constant(64, 1, 0.0);
        bias1.to_device(HANDLE);
        weight2 = Matrix::constant(64, 10, .001);
        weight2.to_device(HANDLE);
        bias2 = Matrix::constant(10, 1, 0.0);
        bias2.to_device(HANDLE);
    }

    FCNN(const std::string &weights_dir)
    {
        weight1 = Matrix::from_npy(weights_dir + "/w1.npy");
        weight1.to_device(HANDLE);
        bias1 = Matrix::from_npy(weights_dir + "/b1.npy");
        bias1.to_device(HANDLE);
        weight2 = Matrix::from_npy(weights_dir + "/w2.npy");
        weight2.to_device(HANDLE);
        bias2 = Matrix::from_npy(weights_dir + "/b2.npy");
        bias2.to_device(HANDLE);
    }

    Matrix operator()(Matrix &input)
    {
        std::vector<cl::Event> events;
        events.resize(3);
        Matrix y;
        std::tie(y, events[0]) = apply_matmul(input, weight1, HANDLE, MATMUL_KERNEL);
        events[1] = events[0];
        events[2] = events[0];
        events[1] = apply_bias(y, bias1, HANDLE, BIAS_RELU6_KERNEL, &events);

        std::tie(y, events[2]) = apply_matmul(y, weight2, HANDLE, MATMUL_KERNEL, &events);
        apply_bias(y, bias2, HANDLE, BIAS_SOFTMAX_KERNEL, &events);
        return y;
    }
};

#endif /* end of include guard: NNONFPGA_NET */
