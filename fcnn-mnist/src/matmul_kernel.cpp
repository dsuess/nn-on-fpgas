#include "matmul_kernel.hpp"
#include <iostream>

extern "C" void matmul_kernel(const float *const matrixA, const float *const matrixB, const uint rowsA, const uint colsA, const uint colsB, float *const out)
{
   for (uint i = 0; i < rowsA; ++i)
   {
      for (uint j = 0; j < colsB; ++j)
      {
         // Nulling result here causes issues when running in hw-emu mode.
         // Looks like io isn't updated "in time"
         const uint io = colsB * i + j;
         for (uint k = 0; k < colsA; ++k)
         {
            const uint ia = colsA * i + k;
            const uint ib = colsB * k + j;
            out[io] += matrixA[ia] * matrixB[ib];
         }
      }
   }
}