import numpy as np
import time
import torch
from tqdm import tqdm
from gifify import gifify
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import linregress
from sorcerun.git_utils import get_repo
import sys

REPO = get_repo()
sys.path.append(f"{REPO.working_dir}")
sys.path.append(f"{REPO.working_dir}/langevin-gpu/python/")

from torch_fvm import FokkerPlanckGraphGPUSolver
from langevin_simulator import LangevinSimulator, STEPS_PER_KERNEL


REAL_STEPS_PER_KERNEL = 1000


def analytical_steady_state_linear(
    num_edges,
    edge_length,
    num_bins,
    drift_coeffs,
    sigma,
):
    """Compute analytical steady state density for a metric graph with linear potential

    Assumes:
        1. all edges have the same length
        2. potential is linear on each edge
        3. diffusion coefficient is constant

        These assumptions are placed only to simplify the computation of the
        steady state density, and does not affect the generality of the method.

    Args:
        num_edges: int, number of edges in the graph
        edge_length: float, length of each edge
        num_bins: int, number of bins to use for density discretization
        drift_coeffs: array of shape (num_edges,), drift coefficients for each edge
        sigma: float, sde noise strength

    Returns:
        x_centers: array of shape (num_bins,), centers of the bins
        p: array of shape (num_edges, num_bins), steady state density

    """
    D = sigma**2 / 2
    bins = np.arange(num_bins + 1) * edge_length / num_bins
    x_centers = (bins[1:] + bins[:-1]) / 2

    p = np.zeros((num_edges, num_bins))
    slopes = drift_coeffs / D

    C = 1 / (1 / abs(slopes)).sum()

    for edge_idx in range(num_edges):
        # Compute steady state density
        p[edge_idx] = C * np.exp(slopes[edge_idx] * x_centers)

    return x_centers, p


def analytical_steady_state_quadratic(
    num_edges,
    edge_length,
    num_bins,
    drift_coeffs,
    sigma,
):
    """Compute analytical steady state density for a metric graph with quadratic potential

    Assumes:
        1. all edges have the same length
        2. potential is quadratic on each edge
        3. diffusion coefficient is constant

        These assumptions are placed only to simplify the computation of the
        steady state density, and does not affect the generality of the method.

    Args:
        num_edges: int, number of edges in the graph
        edge_length: float, length of each edge
        num_bins: int, number of bins to use for density discretization
        drift_coeffs: array of shape (num_edges,), drift coefficients for each edge
        sigma: float, sde noise strength

    Returns:
        x_centers: array of shape (num_bins,), centers of the bins
        p: array of shape (num_edges, num_bins), steady state density

    """
    D = sigma**2 / 2
    bins = np.arange(num_bins + 1) * edge_length / num_bins
    x_centers = (bins[1:] + bins[:-1]) / 2

    p = np.zeros((num_edges, num_bins))

    B = np.sqrt(2 / (D * np.pi)) / (1 / np.sqrt(abs(drift_coeffs))).sum()

    for edge_idx in range(num_edges):
        # Compute steady state density
        p[edge_idx] = B * np.exp(drift_coeffs[edge_idx] / (2 * D) * x_centers**2)

    return x_centers, p


ANALYTIC_SS = {
    "linear": analytical_steady_state_linear,
    "quadratic": analytical_steady_state_quadratic,
}


def log_slope_estimate(y, dx=1e-6):
    # fit least squares line to log(y) use stats linregress

    x = np.arange(len(y)) * dx
    log_y = np.log(y)
    m, _, _, _, _ = linregress(x, log_y)

    return m


def get_error(p1, p2, dx, error_norm):
    """Compute error between two densities on metric graph

    shape of p1, p2: (num_edges, num_bins)
    shape of dx: (num_edges,)

    error_norm: int, norm to use for error computation

    Returns:
        scalar: the error

    """
    error = ((abs(p1 - p2) ** error_norm) * dx[:, None]).sum() ** (1 / error_norm)
    return error


def adapter(config, _run):

    num_particles = config["num_particles"]
    num_bins = config["num_bins"]
    steps = config["steps"]
    dt = config["dt"]
    sigma = config["sigma"]
    num_edges = config["num_edges"]
    edge_lengths = np.array(config["edge_lengths"])
    drift_coeffs = np.array(config["drift_coeffs"])
    jump_weights = np.array(config["jump_weights"])
    potential_type = config["potential_type"]
    make_gif = config["make_gif"]
    error_norm = config["error_norm"]
    run_fvm = config["run_fvm"]
    backend = config["backend"]

    edge_length = np.max(edge_lengths)
    dx = edge_lengths / num_bins

    # check that all edges have same length
    assert np.allclose(
        edge_lengths, edge_length
    ), "Currently all edges must have same length"

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(num_edges)]

    # FVM solver
    if run_fvm:
        print("Initializing FVM solver")
        solver = FokkerPlanckGraphGPUSolver(
            num_edges=num_edges,
            edge_length=edge_length,
            dx=edge_length / num_bins,
            dt=dt,
            sigma=sigma,
            jump_weights=jump_weights,
            drift_coeffs=drift_coeffs,
            potential_type=potential_type,
            device="cuda",
        )
        dt = solver.dt

        # Set initial uniform distribution
        with torch.no_grad():
            # Uniform density: (1/num_edges) / edge_length per cell
            solver.p[:, :] = (1.0 / num_edges) / edge_length

    # Initialize Langevin simulator
    print("Initializing Langevin simulator")
    sim = LangevinSimulator(
        num_particles=num_particles,
        num_edges=num_edges,
        edge_lengths=edge_lengths,
        jump_weights=jump_weights,
        backend=backend,
    )

    # Set initial conditions
    initial_edges = np.random.randint(0, num_edges, size=num_particles, dtype=np.int32)
    initial_positions = np.random.uniform(
        0, min(edge_lengths), size=num_particles
    ).astype(np.float32)

    # Upload initial state
    sim.upload_initial_state(initial_edges, initial_positions)
    normalizing_factors = num_particles * dx[:, None]

    x_centers, p_ss = ANALYTIC_SS[potential_type](
        num_edges=num_edges,
        edge_length=edge_length,
        num_bins=num_bins,
        drift_coeffs=drift_coeffs,
        sigma=sigma,
    )

    # Run simulation
    print("Running simulation")
    t = tqdm(range((steps + STEPS_PER_KERNEL - 1) // STEPS_PER_KERNEL))
    g = (
        gifify(
            t,
            filename=f"{REPO.working_dir}/file_storage/runs/{_run._id}/density.gif",
        )
        if make_gif
        else t
    )
    total = 1
    for i in g:
        start = time.time()
        sim.multi_step(
            base_dt=dt,
            sigma=sigma,
        )
        hists = sim.compute_histograms(num_bins=num_bins) / normalizing_factors
        langevin_time = time.time() - start
        if REAL_STEPS_PER_KERNEL != STEPS_PER_KERNEL:
            langevin_time *= REAL_STEPS_PER_KERNEL / STEPS_PER_KERNEL
        _run.log_scalar("langevin_time", langevin_time, i)

        if run_fvm:
            start = time.time()
            for _ in range(STEPS_PER_KERNEL):
                solver.step()
            _, p = solver.get_density()
            fvm_time = time.time() - start
            _run.log_scalar("fvm_time", fvm_time, i)

            # Get density from FVM solver
            total = solver.check_conservation()

        time_elapsed = (i + 1.0) * STEPS_PER_KERNEL * dt

        # compute densities from Langevin
        bins = np.linspace(0, edge_length, num_bins)

        # check that density integrates to 1
        assert np.allclose(np.sum(hists * dx[:, None]), 1.0), "Density not normalized"

        postfix = {}
        # Compute error
        if run_fvm:
            fvm_langevin_error = get_error(p, hists, dx, error_norm=error_norm)
            _run.log_scalar("fvm_langevin_error", fvm_langevin_error, i)
            fvm_ss_error = get_error(p_ss, p, dx, error_norm=error_norm)
            _run.log_scalar("fvm_ss_error", fvm_ss_error, i)
            postfix.update(
                {
                    "fvm_langevin_error": fvm_langevin_error,
                    "fvm_ss_error": fvm_ss_error,
                    "prob": total,
                }
            )

        langevin_ss_error = get_error(p_ss, hists, dx, error_norm=error_norm)
        _run.log_scalar("langevin_ss_error", langevin_ss_error, i)
        postfix.update(
            {
                "time": time_elapsed,
                "langevin_ss_error": langevin_ss_error,
            }
        )
        t.set_postfix(postfix)

        # print(f"Langevin at 0:\t {hists[:, 0]}")
        # print(f"Steady s at 0:\t {p_ss[:, 0]}")
        # if run_fvm:
        #     print(f"FVM at 0:\t {p[:, 0]}")

        # Plot histogram for each edge
        if make_gif:
            plt.figure(figsize=(12, 6))
            slope_ratio = 0
            for edge_idx in range(num_edges):
                # normalize histogram to get density
                slope_sim = log_slope_estimate(hists[edge_idx][:10], dx=dx[edge_idx])
                plt.plot(
                    x_centers,
                    hists[edge_idx],
                    "-",
                    label=f"Slope={slope_sim:.4f}, Edge {edge_idx} "
                    + f"(L={edge_lengths[edge_idx]:.1f}, "
                    + f"dV = {drift_coeffs[edge_idx]}), particle method",
                    color=colors[edge_idx],
                )
                if run_fvm:
                    slope_solver = log_slope_estimate(p[edge_idx][:10], dx=dx[edge_idx])
                    plt.plot(
                        x_centers,
                        p[edge_idx],
                        "--",
                        label=f"Slope={slope_solver:.4f}, Edge {edge_idx}, solver, ratio={slope_sim/slope_solver:.4f}",
                        color=colors[edge_idx],
                    )
                    slope_ratio += slope_sim / slope_solver

                plt.plot(
                    x_centers,
                    p_ss[edge_idx],
                    "-.",
                    label=f"Edge {edge_idx}, analytical_steady_state",
                    color=colors[edge_idx],
                )

            slope_ratio /= num_edges

            plt.title(f"T = {time_elapsed:.5f}")
            plt.xlabel("Position on Edge")
            plt.ylabel("Normalized Density")

            max_nonzero_x = np.max(bins[np.any(hists > 1e-6, axis=0)])
            plt.xlim(0, max_nonzero_x)
            # plt.xlim(-1e-2, 1.2e-1)
            plt.ylim(1e-7, 20)
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Add text box with simulation parameters
            textstr = "\n".join(
                (
                    f"Particles = {num_particles}",
                    f"σ = {sigma}",
                    f"dt = {dt}",
                    f"Steps = {i * STEPS_PER_KERNEL}",
                    f"Average ratio of slopes = {slope_ratio:.4f}",
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
            # plt.show()
    print("Simulation complete")
    print("Computing bounces")
    bounces, bounce_instances = sim.get_bounces()
    ii = np.where(bounce_instances > 0)
    total_bounces = np.sum(bounces[ii])
    total_instances = np.sum(bounce_instances[ii])
    avg_bounces = np.mean(bounces[ii] / bounce_instances[ii])

    _run.log_scalar("avg_bounces", avg_bounces, i)
    _run.log_scalar("total_bounces", total_bounces, i)
    _run.log_scalar("total_instances", total_instances, i)
    print(f"Average bounces: {avg_bounces}")
    print(f"Total bounces: {total_bounces}")
    print(f"Total instances: {total_instances}")


if __name__ == "__main__":
    from sorcerun.sacred_utils import run_sacred_experiment
    import importlib

    sys.path.append(f"{REPO.working_dir}/main")
    import config

    importlib.reload(config)
    r = run_sacred_experiment(adapter, config.config)

    matplotlib.use("module://matplotlib-backend-sixel")

    runs_dir = f"{REPO.working_dir}/file_storage/runs"
    run_id = r._id
    flame_svg = f"{runs_dir}/{run_id}/profile.svg"

    import cairosvg

    with open(flame_svg, "rb") as f:
        cairosvg.svg2png(
            file_obj=f,
            write_to=f"{runs_dir}/{run_id}/profile.png",
        )

    plt.figure(figsize=(24, 20))
    plt.imshow(plt.imread(f"{runs_dir}/{run_id}/profile.png"))
    plt.axis("off")
    plt.show()
