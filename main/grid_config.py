import numpy as np
from sorcerun.git_utils import (
    is_dirty,
    get_repo,
    get_commit_hash,
    get_time_str,
    get_tree_hash,
)
from config import _compute_stability_limit

repo = get_repo()
commit_hash = get_commit_hash(repo)
time_str = get_time_str()
dirty = is_dirty(repo)


# Configuration
T = 1.0
sigma = 1.0
num_edges = 5
edge_length = 10.0
drift_coeffs = np.array([-10, -20, -30, -40, -50], dtype=np.float32).tolist()
# drift_coeffs = np.array([-30] * num_edges)
make_gif = False
error_norm = 2
run_fvm = False
potential_type = "quadratic"
#
#
edge_lengths = np.array([edge_length] * num_edges).tolist()
jump_weights = np.ones(num_edges, dtype=np.float32)
jump_weights /= jump_weights.sum()
jump_weights = jump_weights.tolist()


# %%
def make_config(num_particles, num_bins, dt, r):
    max_stable_dt = _compute_stability_limit(
        drift_coeffs,
        D=sigma**2 / 2,
        dx=edge_length / num_bins,
    )
    if dt > max_stable_dt and run_fvm:
        print(f"Skipping dt={dt} > {max_stable_dt}")
        return None
    steps = int(T / dt)
    c = {
        "num_particles": num_particles,
        "num_bins": num_bins,
        "steps": steps,
        "dt": dt,
        "sigma": sigma,
        "num_edges": num_edges,
        "edge_lengths": edge_lengths,
        "drift_coeffs": drift_coeffs,
        "jump_weights": jump_weights,
        "make_gif": make_gif,
        "error_norm": error_norm,
        "run_fvm": run_fvm,
        "potential_type": potential_type,
        "backend": "cuda",
        #
        "repeat": r,
        "commit_hash": commit_hash,
        "main_tree_hash": get_tree_hash(repo, "main"),
        "time_str": time_str,
        "dirty": dirty,
    }
    return c


num_repeats = 3
parts = np.logspace(4, 8, 5, base=10, dtype=int).tolist()
bins = np.logspace(2, 6, 5, base=10, dtype=int).tolist()
dts = (1 * np.logspace(-6, -3, 4, base=10)).tolist()

_compute_stability_limit(
    drift_coeffs,
    D=sigma**2 / 2,
    dx=edge_length / max(bins),
)

configs = [
    make_config(
        num_particles=npart,
        num_bins=nbin,
        dt=dt,
        r=r,
    )
    for npart in parts
    for nbin in bins
    for dt in dts
    for r in range(num_repeats)
]

configs = [c for c in configs if c is not None]
print(f"Generated {len(configs)} configurations")
print("Max steps", max([c["steps"] for c in configs]))
print("Min steps", min([c["steps"] for c in configs]))
