"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Constrained random search within a shrinking hyper-sphere centered at origin for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        center = (lo + hi) / 2.0
        radii = (hi - lo) / 2.0

        # WFG9 Pareto front is near the origin in decision space
        # We'll sample within a hyper-sphere centered at origin (0 in transformed space)
        # Transform: x = center + radii * u, where u is in [-1, 1]^d
        # Sphere constraint: ||u|| <= r(t) where r shrinks from 1.0 to ~0.1

        rng = np.random.default_rng(42)

        all_points = []
        all_objs = []

        # Phase 1: Sample within shrinking sphere
        n_total = self.budget
        # Start with a large radius, shrink linearly
        # r(t) = 1.0 * (1 - t/n_total) + 0.05

        for i in range(n_total):
            if time.time() > deadline:
                break

            # Compute current radius
            frac = i / max(n_total - 1, 1)
            r_current = 0.1 + 0.9 * (1.0 - frac)  # shrinks from 1.0 to 0.1

            # Generate random point in hyper-sphere of radius r_current
            # Sample direction uniformly on sphere, then scale by r_current * (uniform^(1/dim))
            while True:
                u = rng.standard_normal(self.dim)
                norm_u = np.linalg.norm(u)
                if norm_u > 0:
                    u = u / norm_u
                    break

            # Radius within sphere: r_current * (U^(1/dim)) for uniform distribution in ball
            scale = r_current * (rng.random() ** (1.0 / self.dim))
            u_scaled = u * scale

            # Transform back to original space
            x = center + radii * u_scaled
            x = np.clip(x, lo, hi)

            f = np.array(problem(x))

            all_points.append(x)
            all_objs.append(f)

        if not all_objs:
            return np.array([])

        return np.array(all_points)
# EVOLVE-BLOCK-END
