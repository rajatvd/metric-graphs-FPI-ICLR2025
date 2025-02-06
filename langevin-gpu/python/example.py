# python/example.py
import numpy as np
from tqdm import tqdm
from gifify import gifify
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use("module://matplotlib-backend-sixel")
from langevin_simulator import LangevinSimulator, STEPS_PER_KERNEL

# Configuration
num_particles = int(1e8)
# num_particles = int(1e6)  # 1 million particles
num_bins = int(1e3)
T = 2e1
dt = 5e-5
steps = int(T / dt)
sigma = 1.0
num_edges = 3
edge_lengths = np.array([10.0] * num_edges)
jump_weights = np.ones(num_edges, dtype=np.float32)
jump_weights /= jump_weights.sum()


def main():
    # Initialize simulator
    sim = LangevinSimulator(
        num_particles=num_particles,
        num_edges=num_edges,
        edge_lengths=edge_lengths,
        jump_weights=jump_weights,
    )

    # Set initial conditions
    initial_edges = np.random.randint(0, num_edges, size=num_particles, dtype=np.int32)
    initial_positions = np.random.uniform(
        0, min(edge_lengths), size=num_particles
    ).astype(np.float32)

    # Upload initial state
    sim.upload_initial_state(initial_edges, initial_positions)

    for i in gifify(tqdm(range(steps // STEPS_PER_KERNEL + 1))):
        sim.multi_step(
            base_dt=dt,
            sigma=sigma,
        )

        # Get results
        # print("Getting results")
        # final_edges, final_positions, bounces = sim.get_state()

        # Plot histograms
        plt.figure(figsize=(12, 6))

        # Create bins based on maximum edge length
        max_length = max(edge_lengths)
        bins = np.linspace(0, max_length, num_bins)
        x = (bins[:-1] + bins[1:]) / 2

        hists = sim.compute_histograms(num_bins=num_bins) / num_particles
        dx = edge_lengths / num_bins
        hists = hists / dx[:, None]

        # check that density integrates to 1

        assert np.allclose(np.sum(hists * dx[:, None]), 1.0)

        # Plot histogram for each edge
        for edge_idx in range(num_edges):
            # normalize histogram to get density
            plt.plot(
                bins,
                hists[edge_idx],
                label=f"Edge {edge_idx} (L={edge_lengths[edge_idx]:.1f}, dV = {(edge_idx+1)*10})",
            )
            # plt.hist(
            #     edge_positions,
            #     bins=bins,
            #     alpha=0.5,
            #     label=f"Edge {edge_idx} (L={edge_lengths[edge_idx]:.1f})",
            #     density=True,
            # )

        plt.title(f"T = {i * STEPS_PER_KERNEL * dt:.5f}")
        plt.xlabel("Position on Edge")
        plt.ylabel("Normalized Density")

        max_nonzero_x = np.max(bins[np.any(hists > 1e-6, axis=0)])
        # plt.xlim(0, max_nonzero_x)
        plt.xlim(0, 1.2)
        plt.ylim(1e-6, 20)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add text box with simulation parameters
        textstr = "\n".join(
            (
                f"Particles = {num_particles}",
                f"σ = {sigma}",
                f"dt = {dt}",
                f"Steps = {i * STEPS_PER_KERNEL}",
            )
        )
        plt.gca().text(
            0.95,
            0.95,
            textstr,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

        plt.yscale("log")
        plt.tight_layout()
        # plt.show()


if __name__ == "__main__":
    main()
