import numpy as np
from sorcerun.git_utils import (
    is_dirty,
    get_repo,
    get_commit_hash,
    get_time_str,
    get_tree_hash,
)

repo = get_repo()


def _compute_stability_limit(drift_coeffs, D, dx):
    """Calculate CFL conditions"""
    v_max = np.max(np.abs(drift_coeffs))
    dt_adv = dx / v_max if v_max > 0 else float("inf")
    dt_diff = (dx**2) / (2 * D)
    max_stable_dt = min(dt_adv, dt_diff) / 2
    return max_stable_dt


# Configuration
T = 2
num_particles = int(1e6)
num_bins = int(1e3)
sigma = 1.0
num_edges = 5
edge_length = 10.0
drift_coeffs = np.array([-10, -20, -30, -40, -50], dtype=np.float32)
potential_type = "quadratic"
# drift_coeffs = np.array([-30] * num_edges)
make_gif = True
error_norm = 2
run_fvm = True
#
#
edge_lengths = np.array([edge_length] * num_edges)
jump_weights = np.ones(num_edges, dtype=np.float32)
jump_weights /= jump_weights.sum()

max_stable_dt = _compute_stability_limit(
    drift_coeffs,
    D=sigma**2 / 2,
    dx=edge_length / num_bins,
)

dt = max_stable_dt
print(f"Max stable dt: {max_stable_dt}")
dt = 1e-5
steps = int(T / dt)

config = {
    "num_particles": num_particles,
    "num_bins": num_bins,
    "steps": steps,
    "dt": dt,
    "sigma": sigma,
    "num_edges": num_edges,
    "edge_lengths": edge_lengths,
    "drift_coeffs": drift_coeffs,
    "potential_type": potential_type,
    "jump_weights": jump_weights,
    "make_gif": make_gif,
    "error_norm": error_norm,
    "run_fvm": run_fvm,
    "backend": "cuda",
    #
    "commit_hash": get_commit_hash(repo),
    "main_tree_hash": get_tree_hash(repo, "main"),
    "time_str": get_time_str(),
    "dirty": is_dirty(get_repo()),
}
