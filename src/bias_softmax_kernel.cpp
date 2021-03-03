typedef unsigned int uint;
#include "hls_math.h"

extern "C" void bias_softmax_kernel(float *const activation, const float *const bias, const uint batch_size, const uint dim)
{
   for (uint b = 0; b < batch_size; b++)
   {
      float accum = 0.;
      for (uint d = 0; d < dim; d++)
      {
         const uint ia = dim * b + d;
         activation[ia] = exp(activation[ia] + bias[d]);
         accum += activation[ia];
      }
      for (uint d = 0; d < dim; d++)
      {
         const uint ia = dim * b + d;
         activation[ia] /= accum;
      }
   }
}