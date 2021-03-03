#include <iostream>
typedef unsigned int uint;

inline float relu6(const float x)
{
   if (x < 0.f)
      return 0.f;
   if (x > 6.f)
      return 6.f;
   return x;
}

extern "C" void bias_relu6_kernel(float *const activation, const float *const bias, const uint batch_size, const uint dim)
{
   for (uint b = 0; b < batch_size; b++)
   {
      for (uint d = 0; d < dim; d++)
      {
         const uint ia = dim * b + d;
         activation[ia] = relu6(activation[ia] + bias[d]);
      }
   }
}