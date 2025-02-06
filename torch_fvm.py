import torch
import numpy as np

TOL = 1e-6


class FokkerPlanckGraphGPUSolver:
    def __init__(
        self,
        num_edges=3,
        edge_length=1.0,
        dx=0.1,
        dt=0.1,
        sigma=1.0,
        jump_weights=None,
        drift_coeffs=None,
        potential_type="linear",
        device="cuda",
    ):
        # Simulation parameters
        self.num_edges = num_edges
        self.edge_length = edge_length
        self.dx = dx
        self.dt = dt
        self.D = sigma**2 / 2
        self.device = device

        # Discretization
        self.N = int(edge_length / dx)
        self.x = torch.arange(0, self.N, device=device) * dx + dx / 2
        assert self.x.shape[0] == self.N

        # Graph parameters
        self.jump_weights = torch.tensor(
            jump_weights,
            dtype=torch.float32,
            device=device,
        )
        self.drift_coeffs = torch.tensor(
            drift_coeffs,
            dtype=torch.float32,
            device=device,
        )

        # Initialize density (probability density per unit length)
        self.p = torch.full(
            (num_edges, self.N), (1.0 / num_edges) / edge_length, device=device
        )

        # Index arrays for vectorized operations
        self.left_idx = torch.arange(-1, self.N, device=device)
        self.left_idx[0] = 0  # Handle vertex boundary
        self.right_idx = torch.arange(1, self.N + 1, device=device)
        self.right_idx[-1] = self.N - 1  # Handle edge end boundary

        self.potential_type = potential_type
        if self.potential_type == "linear":
            self.drifts = self.drift_coeffs[:, None] * torch.ones(
                (1, self.N), device=self.device
            )
        elif self.potential_type == "quadratic":
            self.drifts = self.drift_coeffs[:, None] * self.x[None, :]

        # Precompute stability criteria
        self._compute_stability_limit()
        self.check_stability()

    def _compute_stability_limit(self):
        """Calculate CFL conditions"""
        v_max = torch.max(torch.abs(self.drifts)).item()
        self.dt_adv = self.dx / v_max if v_max > 0 else float("inf")
        self.dt_diff = (self.dx**2) / (2 * self.D)
        self.max_stable_dt = min(self.dt_adv, self.dt_diff) / 2

    def check_stability(self):
        """Verify CFL conditions are satisfied"""
        if self.dt is None:
            print(
                f"No time step specified,"
                + f" using maximum stable dt {self.max_stable_dt:.2e}"
            )
            self.dt = self.max_stable_dt
        if self.dt > self.max_stable_dt:
            raise ValueError(
                f"Time step {self.dt} exceeds stability limit {self.max_stable_dt:.2e}"
            )
        else:
            print(
                f"Stability check passed: dt = {self.dt:.2e}, max dt = {self.max_stable_dt:.2e}"
            )

    def step(self):
        # Initialize fluxes with zero (including boundary fluxes)
        fluxes = torch.zeros((self.num_edges, self.N + 1), device=self.device)

        # Vertex flux calculations
        for source in range(self.num_edges):
            real_source_density = self.p[source, 0] / self.jump_weights[source]

            # Diffusion flux between edges
            for target in range(self.num_edges):
                if target == source:
                    continue

                real_target_density = self.p[target, 0] / self.jump_weights[target]
                real_diffusion_flux = (
                    self.D * (real_source_density - real_target_density) / self.dx
                ) / 2
                # real_diffusion_flux /= self.num_edges - 1
                fluxes[source, 0] -= real_diffusion_flux * self.jump_weights[source]
                fluxes[target, 0] += real_diffusion_flux * self.jump_weights[target]

            # Drift flux handling using precomputed drift coefficients
            drift = self.drift_coeffs[source]  # Now using direct drift coefficients
            if drift < 0:
                real_drift_flux = -drift * real_source_density
                fluxes[source, 0] -= real_drift_flux * self.jump_weights[source]

                # Distribute drift flux according to jump weights
                for target in range(self.num_edges):
                    if target != source:
                        fluxes[target, 0] += (
                            real_drift_flux
                            / (self.num_edges - 1)
                            * self.jump_weights[target]
                        )

        # Flux balance checks (maintained from original)
        weighted_flux_sum = torch.sum(fluxes[:, 0] * self.jump_weights)
        # if torch.abs(weighted_flux_sum) > TOL:
        #     print(f"Weighted flux imbalance: {weighted_flux_sum.item():.2e}")

        unweighted_flux_sum = torch.sum(fluxes[:, 0])
        # if torch.abs(unweighted_flux_sum) > TOL:
        #     print(f"Unweighted flux imbalance: {unweighted_flux_sum.item():.2e}")

        # Interior flux calculations
        for edge in range(self.num_edges):
            drifts = self.drifts[edge]
            signed_drift_fluxes = drifts * self.p[edge]

            # Upwind scheme implementation
            advection_flux = torch.zeros(self.N + 1, device=self.device)
            right_moving = torch.where(drifts > 0)[0]
            advection_flux[right_moving + 1] = signed_drift_fluxes[right_moving]
            left_moving = torch.where(drifts < 0)[0]
            advection_flux[left_moving] += signed_drift_fluxes[left_moving]

            # Diffusion fluxes (central difference remains same)
            diffusion_flux = -self.D * (self.p[edge, 1:] - self.p[edge, :-1]) / self.dx

            # Combine fluxes (ignore first and last flux positions)
            fluxes[edge, 1:-1] = advection_flux[1:-1] + diffusion_flux

        # Update cell densities (same as before)
        for edge in range(self.num_edges):
            flux_diff = fluxes[edge, 1:] - fluxes[edge, :-1]
            self.p[edge] -= self.dt * flux_diff / self.dx

        return self.p

    def get_density(self):
        """Return density distribution as numpy arrays"""
        return self.x.cpu().numpy(), self.p.cpu().numpy()

    def check_conservation(self):
        """Verify total probability conservation (should be ~1)"""
        total = torch.sum(self.p * self.dx).item()
        return total


if __name__ == "__main__":
    # Parameters matching particle simulation
    params = {
        "num_edges": 3,
        "edge_length": 1.0,
        "dx": 0.1,
        "dt": 0.1,
        "sigma": 1.0,
        "jump_weights": [0.3, 0.3, 0.4],
        "drift_coeffs": [-10, -20, -30],  # -dV coefficients
        "device": "cuda",
    }

    solver = FokkerPlanckGraphGPUSolver(**params)

    # Run simulation
    num_steps = 10
    for _ in range(num_steps):
        solver.step()

    # Get results
    x, p = solver.get_density()

    print("Final density distribution:")
    for i in range(params["num_edges"]):
        print(f"Edge {i}: {p[i].round(4)}")

    print("\nTotal probability:", np.sum(p).round(6))
