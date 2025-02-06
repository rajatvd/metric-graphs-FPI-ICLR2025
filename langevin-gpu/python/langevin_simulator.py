import numpy as np
import torch
import pycuda.autoinit
from tqdm import tqdm
from pycuda.compiler import SourceModule, DEFAULT_NVCC_FLAGS
import pycuda.driver as cuda
from sorcerun.git_utils import get_repo

STEPS_PER_KERNEL = 1000
REPO = get_repo()
TOL = 1e-6


# %%
def torch_random_choice(a, size=1, replace=True, p=None, device=None):
    # If 'a' is an integer, create a tensor from 0 to a-1
    if isinstance(a, int):
        population = torch.arange(a, device=device, dtype=torch.int32)
    else:
        population = a
    num_elements = len(population)

    if p is None:
        # Uniform sampling
        if replace:
            indices = torch.randint(num_elements, (size,), device=device)
        else:
            indices = torch.randperm(num_elements)[:size]
    else:
        # Normalize probabilities
        p = torch.tensor(p, dtype=torch.float, device=device)
        p = p / p.sum()
        # Sample using multinomial
        indices = torch.zeros(size, dtype=torch.int64, device=device)
        if replace:
            torch.multinomial(p, size, replacement=True, out=indices)
        else:
            torch.multinomial(p, size, replacement=False, out=indices)

    result = population[indices]
    # result.type()
    # import ipdb; ipdb.set_trace() # fmt: skip
    return result


# %%
def solve_quadratic_vectorized_torch(a, b, c, sign, print_info=False):
    out = torch.zeros_like(c, device=a.device)
    # print(a, b, c)

    zeroa = a == 0
    out[zeroa] = -c[zeroa] / b[zeroa]
    positive_b = (b >= 0) & ~zeroa
    out[positive_b] = (
        -b[positive_b]
        + sign * torch.sqrt(b[positive_b] ** 2 - 4 * a[positive_b] * c[positive_b])
    ) / (2 * a[positive_b])

    negative_b = (b < 0) & ~zeroa
    out[negative_b] = (2 * c[negative_b]) / (
        -b[negative_b]
        - sign * torch.sqrt(b[negative_b] ** 2 - 4 * a[negative_b] * c[negative_b])
    )
    # if zeroa.any():
    #     import ipdb; ipdb.set_trace()  # fmt: skip

    return out


def quad_drift(i, x):
    return -(10.0 * (i + 1.0)) * x


def lin_drift(i, x):
    return -(10.0 * (i + 1.0))


# %%
class LangevinSimulator:
    def __init__(
        self,
        num_particles,
        num_edges,
        edge_lengths,
        jump_weights,
        backend="cuda",
    ):
        self.num_particles = num_particles
        self.num_edges = num_edges
        self.backend = backend
        self.edge_lengths_host = np.array(edge_lengths, dtype=np.float32)

        if backend == "cuda":
            with open(f"{REPO.working_dir}/langevin-gpu/lib/kernel.cubin", "rb") as f:
                cubin = f.read()

            self.mod = cuda.module_from_buffer(cubin)

            # Get kernel function
            # self.kernel = self.mod.get_function("langevin_kernel")
            print("Loading CUDA kernels")
            self.setup_kernel = self.mod.get_function("setup_kernel")
            self.multi_step_kernel = self.mod.get_function("langevin_multi_step_kernel")
            self.compute_hist = self.mod.get_function("compute_histogram_kernel")

            # Allocate device memory
            print("Allocating device memory")
            self.edges = cuda.mem_alloc(num_particles * np.int32().nbytes)
            self.positions = cuda.mem_alloc(num_particles * np.float32().nbytes)
            self.bounces = cuda.mem_alloc(num_particles * np.int32().nbytes)
            self.bounce_instances = cuda.mem_alloc(num_particles * np.int32().nbytes)
            self.edge_lengths = cuda.to_device(self.edge_lengths_host)
            cum_weights = np.array(jump_weights, dtype=np.float32).cumsum()
            print(cum_weights)
            self.jump_weights = cuda.to_device(cum_weights)

            # Allocating RNG states
            print("Allocating RNG states")
            self.states = cuda.mem_alloc(num_particles * 48)  # sizeof(curandState)

        elif backend == "torch":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device == "cpu":
                raise ValueError("Torch backend only supports CUDA")

            self.edges = torch.zeros(
                num_particles, dtype=torch.int32, device=self.device
            )
            self.positions = torch.zeros(
                num_particles, dtype=torch.float32, device=self.device
            )
            self.bounces = torch.zeros(
                num_particles, dtype=torch.int32, device=self.device
            )
            self.bounce_instances = torch.zeros(
                num_particles, dtype=torch.int32, device=self.device
            )
            self.edge_lengths = torch.tensor(
                edge_lengths, dtype=torch.float32, device=self.device
            )
            self.jump_weights = torch.tensor(
                jump_weights, dtype=torch.float32, device=self.device
            )

        else:
            raise ValueError(f"Backend {backend} not supported")

    def get_compute_capability(self):
        """Get GPU compute capability version"""
        dev = pycuda.autoinit.device
        return f"{dev.compute_capability()[0]}{dev.compute_capability()[1]}"

    def multi_step(self, base_dt, sigma):
        if self.backend == "cuda":
            block = (512, 1, 1)
            grid = (int((self.num_particles + block[0] - 1) // block[0]), 1)

            self.multi_step_kernel(
                self.edges,
                self.positions,
                self.bounces,
                self.bounce_instances,
                self.edge_lengths,
                self.jump_weights,
                np.float32(base_dt),
                np.float32(sigma),
                np.int32(self.num_edges),
                np.int32(self.num_particles),
                self.states,
                block=block,
                grid=grid,
            )
            cuda.Context.get_current().synchronize()

        elif self.backend == "torch":
            with torch.no_grad():
                for _ in range(STEPS_PER_KERNEL):
                    self._torch_step(base_dt, sigma)
        else:
            raise ValueError(f"Backend {self.backend} not supported")

    @torch.compile
    def _torch_step(self, base_dt, sigma):

        i = self.edges
        x = self.positions
        edge_lengths = self.edge_lengths

        dt = torch.ones_like(x, device=self.device) * base_dt
        mask = torch.where(dt > 0)[0]
        num_particles = self.num_particles

        while num_particles > 0:

            w = torch.randn(num_particles, device=self.device)
            while torch.any(w == 0):
                # apparantly torch.randn can return 0 exactly
                w = torch.randn(num_particles, device=self.device)
            w[x[mask] == 0] = torch.abs(w[x[mask] == 0])
            # drift = quad_drift(i[mask], x[mask])
            drift = lin_drift(i[mask], x[mask])
            # print(f"drift = {drift}")
            # print(f"i = {i[mask]}")
            # print(f"x = {x[mask]}")
            # print(f"num_particles = {num_particles}")
            # print("-----------------")

            x_next = x[mask] + dt[mask] * drift + sigma * torch.sqrt(dt[mask]) * w

            # particles that have not crossed the vertex
            not_crossed = (x_next > 0) & (x_next < edge_lengths[i[mask]])
            x[mask[not_crossed]] = x_next[not_crossed]
            dt[mask[not_crossed]] = 0

            # particles that have hit the end of the edge
            edge_reflected = x_next >= edge_lengths[i[mask]]
            x[mask[edge_reflected]] = (
                2 * edge_lengths[i[mask[edge_reflected]]] - x_next[edge_reflected]
            )
            dt[mask[edge_reflected]] = 0

            assert torch.all(
                x[mask] >= 0
            ), "x_new < 0 for particles that hit the end of the edge"

            # particles that have crossed the vertex
            crossed = x_next <= 0

            # only increment bounces for particles that start from 0
            self.bounces[(mask & (x[mask] == 0))[crossed]] += 1
            # self.bounce_instances[mask[crossed]] += 1

            # particles that have crossed the vertex and are in the bounce loop

            # import ipdb; ipdb.set_trace()  # fmt: skip

            aa = drift[crossed] * dt[mask[crossed]]
            bb = sigma * torch.sqrt(dt[mask[crossed]]) * w[crossed]
            cc = x[mask[crossed]]
            sqrt_alpha = solve_quadratic_vectorized_torch(
                aa, bb, cc, -1, print_info=False
            )
            alpha = sqrt_alpha**2

            # import ipdb; ipdb.set_trace()  # fmt: skip
            assert torch.all(
                (alpha >= 0) & (alpha <= 1 + TOL)
            ), f"alpha = {alpha} not in [0, 1]"

            expected_zero = (
                x[mask[crossed]]
                + alpha * drift[crossed] * dt[mask[crossed]]
                + sigma * torch.sqrt(alpha * dt[mask[crossed]]) * w[crossed]
            )
            assert torch.all(
                abs(expected_zero) < TOL
            ), f"expected_zero = {expected_zero} not zero, a = {aa}, b = {bb}, c = {cc}"

            dt[mask[crossed]] = (1 - alpha) * dt[mask[crossed]]

            num = int(torch.sum(crossed).item())
            if num > 0:
                i[mask[crossed]] = torch_random_choice(
                    self.num_edges,
                    size=int(torch.sum(crossed).item()),
                    p=self.jump_weights,
                    device=self.device,
                )
            x[mask[crossed]] = 0
            mask = torch.where(dt > 0)[0]
            num_particles = mask.shape[0]

            assert torch.all(x >= 0), "x < 0 after a bounce"

    def _launch_setup_kernel(self):
        """properly launch setup kernel with all required parameters"""
        block = (512, 1, 1)
        grid = (int((self.num_particles + block[0] - 1) // block[0]), 1)

        # generate random seed
        seed = np.random.randint(0, 2**63 - 1, dtype=np.uint64)

        # launch with all 3 parameters
        self.setup_kernel(
            self.states,  # device pointer (p)
            np.ulonglong(seed),  # uint64 (q)
            np.int32(self.num_particles),  # int32 (i)
            block=block,
            grid=grid,
        )

        cuda.Context.get_current().synchronize()

    def get_state(self):
        edges = np.empty(self.num_particles, dtype=np.int32)
        positions = np.empty(self.num_particles, dtype=np.float32)
        bounces = np.empty(self.num_particles, dtype=np.int32)
        bounce_instances = np.empty(self.num_particles, dtype=np.int32)

        cuda.memcpy_dtoh(edges, self.edges)
        cuda.memcpy_dtoh(positions, self.positions)
        cuda.memcpy_dtoh(bounces, self.bounces)
        cuda.memcpy_dtoh(bounce_instances, self.bounce_instances)

        return edges, positions, bounces, bounce_instances

    def get_bounces(self):
        if self.backend == "torch":
            return self.bounces.cpu().numpy(), self.bounce_instances.cpu().numpy()
        elif self.backend == "cuda":
            bounces = np.empty(self.num_particles, dtype=np.int32)
            bounce_instances = np.empty(self.num_particles, dtype=np.int32)

            cuda.memcpy_dtoh(bounces, self.bounces)
            cuda.memcpy_dtoh(bounce_instances, self.bounce_instances)

            return bounces, bounce_instances
        else:
            raise ValueError(f"Backend {self.backend} not supported")

    # Add to langevin_simulator.py
    def upload_initial_state(self, initial_edges, initial_positions):
        """
        Upload initial particle states to the GPU
        Parameters:
            initial_edges: numpy array (int32) - starting edge indices [0, num_edges)
            initial_positions: numpy array (float32) - starting positions [0, edge_length)
        """
        # Validate inputs
        print("Validating initial state")
        if len(initial_edges) != self.num_particles:
            raise ValueError(f"initial_edges must have length {self.num_particles}")
        if len(initial_positions) != self.num_particles:
            raise ValueError(f"initial_positions must have length {self.num_particles}")

        # Convert to proper types
        edges = initial_edges.astype(np.int32)
        positions = initial_positions.astype(np.float32)

        # Check position validity
        for edge_idx in range(self.num_edges):
            edge_mask = edges == edge_idx
            max_pos = self.edge_lengths_host[edge_idx]
            if np.any(positions[edge_mask] < 0) or np.any(
                positions[edge_mask] > max_pos
            ):
                invalid = np.where((positions < 0) | (positions > max_pos))[0]
                raise ValueError(
                    f"Positions out of bounds for edge {edge_idx} "
                    f"(max {max_pos:.2f}) at indices: {invalid[:10]}"
                )

        # Copy to device
        if self.backend == "cuda":
            print("Copying initial state to device")
            cuda.memcpy_htod(self.edges, edges)
            cuda.memcpy_htod(self.positions, positions)

            bounces = np.zeros(self.num_particles, dtype=np.int32)
            cuda.memcpy_htod(self.bounces, bounces)

            print("Initializing RNG states")
            self._launch_setup_kernel()  # Reinitialize RNG after new state
        elif self.backend == "torch":
            self.edges.copy_(torch.tensor(edges, dtype=torch.int32, device=self.device))
            self.positions.copy_(
                torch.tensor(positions, dtype=torch.float32, device=self.device)
            )
        else:
            raise ValueError(f"Backend {self.backend} not supported")

    def compute_histograms(self, num_bins=20):
        """Compute histograms using precompiled CUDA kernel"""

        if self.backend == "cuda":
            block = (512, 1, 1)
            grid = (int((self.num_particles + block[0] - 1) // block[0]), 1)
            # Create output array
            histograms = cuda.to_device(
                np.zeros(self.num_edges * num_bins, dtype=np.int32)
            )

            np_hist = np.empty(self.num_edges * num_bins, dtype=np.int32)
            # Launch kernel
            self.compute_hist(
                self.edges,
                self.positions,
                self.edge_lengths,
                histograms,
                np.int32(num_bins),
                np.int32(self.num_edges),
                np.int32(self.num_particles),
                block=block,
                grid=grid,
            )

            # Copy results to CPU
            cuda.memcpy_dtoh(np_hist, histograms)
            return np_hist.reshape(self.num_edges, num_bins).astype(np.float32)

        elif self.backend == "torch":
            histograms = torch.zeros(
                self.num_edges, num_bins, dtype=torch.int32, device=self.device
            )
            for edge_idx in range(self.num_edges):
                edge_mask = self.edges == edge_idx
                hist = torch.histc(
                    self.positions[edge_mask],
                    bins=num_bins,
                    min=0,
                    max=self.edge_lengths_host[edge_idx],
                )
                histograms[edge_idx] = hist

            return histograms.cpu().numpy()
        else:
            raise ValueError("Torch backend not supported for compute_histograms")
