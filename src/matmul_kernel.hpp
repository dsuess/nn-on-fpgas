#include <tuple>

typedef unsigned int uint;

extern "C" void matmul_kernel(
    const float *const matrixA, const float *const matrixB, const uint rowsA, const uint colsA, const uint colsB, float *const out);
