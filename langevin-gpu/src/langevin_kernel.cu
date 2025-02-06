// langevin-gpu/src/langevin_kernel.cu
#include "langevin_kernel.h"
#include <curand_kernel.h>

#define TOL 1e-10f
#define MAX_ITERATIONS_PER_STEP 100
#define STEPS_PER_KERNEL 1000

extern "C" {
__device__ float device_dV(int edge_index, float x) {
  return 10.0f * (edge_index + 1.0f) * x;
  // return 30.0f;
}

__device__ float solve_quadratic(float A, float B, float C) {
  // Numerically stable solution to quadratic equation
  if (A == 0.0f) {
    return -C / B;
  }
  float discriminant = sqrtf(fmaxf(B * B - 4.0f * A * C, 0.0f));
  if (B > 0.0f) {
    return (-B - discriminant) / (2.0f * A);
  } else {
    return (2.0f * C) / (-B + discriminant);
  }
}

__global__ void
langevin_multi_step_kernel(int *edges, float *positions, int *bounces,
                           int *bounce_instances, const float *edge_lengths,
                           const float *jump_weights, const float base_dt,
                           const float sigma, const int num_edges,
                           const int num_particles, curandState *states) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_particles)
    return;

  // printf("num_particles %d", num_particles);
  int edge = edges[tid];
  float x = positions[tid];
  int bounce_count = bounces[tid];
  int bounce_instance = bounce_instances[tid];
  curandState local_state = states[tid];

  if (edge < 0 || edge >= num_edges) {
    printf("Invalid initial edge %d for particle %d\n", edge, tid);
    edge = 0;
  }

  for (int step = 0; step < STEPS_PER_KERNEL; ++step) {
    float dt = base_dt;
    int iterations = 0;

    while (dt > 0.0f && iterations++ < MAX_ITERATIONS_PER_STEP) {
      float w = curand_normal(&local_state);
      if (x == 0.0f)
        w = fabsf(w);

      float drift = -device_dV(edge, x);
      float sqrt_dt = sqrtf(dt);
      float x_next = x + dt * drift + sigma * sqrt_dt * w;
      float current_length = edge_lengths[edge];

      if (current_length <= 0.0f) {
        printf("Invalid edge length %f for edge %d\n", current_length, edge);
        current_length = 1.0f;
      }

      if (x_next > 0.0f && x_next <= current_length) {
        // no bounce
        x = x_next;
        dt = 0.0f;
      } else if (x_next > current_length) {
        // also no bounce
        x = 2.0f * current_length - x_next;
        dt = 0.0f;
      } else {
        // x_next < 0.0f -- bounce
        if (x != 0.0f) {
          // first bounce
          bounce_instance++;
        }
        bounce_count++;

        float a = drift * dt;
        float b = sigma * sqrt_dt * w;
        float sqrt_alpha = solve_quadratic(a, b, x);
        float alpha = sqrt_alpha * sqrt_alpha;

        dt *= (1.0f - alpha);
        float rand_val = curand_uniform(&local_state);
        int new_edge = 0;
        while (new_edge < num_edges - 1 && rand_val > jump_weights[new_edge]) {
          new_edge++;
        }

        if (new_edge < 0 || new_edge >= num_edges) {
          printf("Invalid new_edge %d, clamping to 0\n", new_edge);
          new_edge = 0;
        }

        edge = new_edge;
        x = 0.0f;
      }
    }
  }

  edges[tid] = edge;
  positions[tid] = x;
  bounces[tid] = bounce_count;
  bounce_instances[tid] = bounce_instance;
  states[tid] = local_state;
}

__global__ void setup_kernel(curandState *states, unsigned long long seed,
                             int num_particles) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_particles)
    return;
  curand_init(seed + tid, 0, 0, &states[tid]);
}

// New histogram kernel
__global__ void compute_histogram_kernel(const int *edges,
                                         const float *positions,
                                         const float *edge_lengths,
                                         int *histograms, int num_bins,
                                         int num_edges, int num_particles) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_particles)
    return;

  const int edge = edges[tid];
  const float pos = positions[tid];

  // Validate input
  if (edge < 0 || edge >= num_edges)
    return;
  const float length = edge_lengths[edge];
  if (pos < 0.0f || pos > length)
    return;

  // Calculate normalized position [0,1]
  const float normalized_pos = pos / length;

  // Determine bin index
  int bin = (int)(normalized_pos * num_bins);
  bin = max(0, min(bin, num_bins - 1)); // Clamp to valid range

  // Atomic increment using 2D indexing: [edge][bin]
  atomicAdd(&histograms[edge * num_bins + bin], 1);
}
}
