#include "matmul_kernel.hpp"
#include <iostream>

extern "C" void matmul_kernel(const float *const matrixA, const float *const matrixB, const uint rowsA, const uint colsA, const uint colsB, float *const out)
{
   for (uint i = 0; i < rowsA; i++)
   {
      for (uint j = 0; j < colsB; j++)
      {
         const auto io = colsB * i + j;
         out[io] = 0.0;
         for (uint k = 0; k < colsA; k++)
         {
            const auto ia = colsA * i + k;
            const auto ib = colsB * k + j;
            out[io] += matrixA[ia] * matrixB[ib];
         }
      }
   }
}