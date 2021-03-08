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
    mat.to_device();
    finish_cl_queue();

    auto result = std::get<0>(apply_matmul(mat, mat, MATMUL_KERNEL));

    finish_cl_queue();
    result.to_cpu();
    finish_cl_queue();

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
    mat.to_device();

    Matrix bias(2, 1);
    bias(0, 0) = 1;
    bias(1, 0) = 2;
    bias.to_device();
    finish_cl_queue();

    apply_bias(mat, bias, BIAS_SOFTMAX_KERNEL);

    finish_cl_queue();
    mat.to_cpu();
    finish_cl_queue();

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
    mat.to_device();

    Matrix bias(2, 1);
    bias(0, 0) = 2;
    bias(1, 0) = 3;
    bias.to_device();
    finish_cl_queue();

    apply_bias(mat, bias, BIAS_RELU6_KERNEL);

    finish_cl_queue();
    mat.to_cpu(HANDLE);
    finish_cl_queue();

    ASSERT_FLOAT_EQ(mat(0, 0), 3);
    ASSERT_FLOAT_EQ(mat(0, 1), 5);
    ASSERT_FLOAT_EQ(mat(1, 0), 5);
    ASSERT_FLOAT_EQ(mat(1, 1), 6);
}
int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    init_kernels();
    return RUN_ALL_TESTS();
}