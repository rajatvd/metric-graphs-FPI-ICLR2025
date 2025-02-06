#ifndef LANGEVIN_KERNEL_H
#define LANGEVIN_KERNEL_H

#include <cuda_runtime.h>

struct SimParams {
  float dt;
  float sigma;
  int num_edges;
};

#endif
