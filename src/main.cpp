#include <string.h>
#include <sys/time.h>

#include <algorithm>
#include <iostream>
#include <tuple>

#include "xcl2.hpp"
#include "matrix.hpp"
#include "net.hpp"

int main(int argc, const char *argv[]) {
  init_kernels();

  auto model = FCNN("../weights/");
  auto input = Matrix::from_npy("../weights/samples.npy");
  input.to_device();
  auto result = model(input);

  finish_cl_queue();
  result.to_cpu();
  finish_cl_queue();

  // print argmax result
  for (int i = 0; i < result.rows; i++) {
    float minval = -1;
    int idx = -1;

    for (int j = 0; j < result.cols; j++) {
      auto val = result(i, j);
      if (minval < val) {
        idx = j;
        minval = val;
      }
    }

    std::cout << idx << " ";
  }
  std::cout << std::endl;
}
