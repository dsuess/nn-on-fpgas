#include <iostream>
#include <tuple>
#include <iostream>
#include "gtest/gtest.h"

#include "utils.hpp"
#include "matrix.hpp"

TEST(KernelTest, MatmulCorrect)
{
    Matrix mat(2, 2);
    mat(0, 0) = 1;
    mat(0, 1) = 2;
    mat(1, 0) = 3;
    mat(1, 1) = 4;
    mat.to_device(HANDLE);
    HANDLE.q.finish();

    auto result = std::get<0>(apply_matmul(mat, mat, HANDLE, MATMUL_KERNEL));

    HANDLE.q.finish();
    result.to_cpu(HANDLE);
    HANDLE.q.finish();

    ASSERT_FLOAT_EQ(result(0, 0), 7.);
    ASSERT_FLOAT_EQ(result(0, 1), 10.);
    ASSERT_FLOAT_EQ(result(1, 0), 15.);
    ASSERT_FLOAT_EQ(result(1, 1), 22.);
}

TEST(KernelTest, BiasSoftmaxCorrect)
{
    Matrix mat(2, 2);
    mat(0, 0) = 1;
    mat(0, 1) = 2;
    mat(1, 0) = 3;
    mat(1, 1) = 4;
    mat.to_device(HANDLE);

    Matrix bias(2, 1);
    bias(0, 0) = 1;
    bias(1, 0) = 2;
    bias.to_device(HANDLE);
    HANDLE.q.finish();

    apply_bias(mat, bias, HANDLE, BIAS_SOFTMAX_KERNEL);

    HANDLE.q.finish();
    mat.to_cpu(HANDLE);
    HANDLE.q.finish();

    ASSERT_FLOAT_EQ(mat(0, 0), 0.11920293);
    ASSERT_FLOAT_EQ(mat(0, 1), 0.88079709);
    ASSERT_FLOAT_EQ(mat(1, 0), 0.11920293);
    ASSERT_FLOAT_EQ(mat(1, 1), 0.88079709);
}

TEST(KernelTest, BiasRelu6Kernel)
{
    Matrix mat(2, 2);
    mat(0, 0) = 1;
    mat(0, 1) = 2;
    mat(1, 0) = 3;
    mat(1, 1) = 4;
    mat.to_device(HANDLE);

    Matrix bias(2, 1);
    bias(0, 0) = 2;
    bias(1, 0) = 3;
    bias.to_device(HANDLE);
    HANDLE.q.finish();

    apply_bias(mat, bias, HANDLE, BIAS_RELU6_KERNEL);

    HANDLE.q.finish();
    mat.to_cpu(HANDLE);
    HANDLE.q.finish();

    ASSERT_FLOAT_EQ(mat(0, 0), 3);
    ASSERT_FLOAT_EQ(mat(0, 1), 5);
    ASSERT_FLOAT_EQ(mat(1, 0), 5);
    ASSERT_FLOAT_EQ(mat(1, 1), 6);
}
int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    DeviceHandle handle = setup_handle();
    init_kernels(handle);
    return RUN_ALL_TESTS();
}