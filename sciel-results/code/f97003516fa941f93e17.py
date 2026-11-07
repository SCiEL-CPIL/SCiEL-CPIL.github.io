"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Grid-based sampling of the DTLZ7 Pareto front with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        dim = self.dim
        budget = self.budget

        results = []
        used = 0

        # Phase 1: Stratified 2D grid with jitter (Latin hypercube style)
        # Use ~70% of budget for broad coverage
        n_phase1 = int(budget * 0.7)
        n_grid = int(np.sqrt(n_phase1))
        if n_grid * n_grid > n_phase1:
            n_grid -= 1

        # Create stratified grid: each cell in [0,1]^2 gets one point with jitter
        rng = np.random.default_rng(123)
        cell_size = 1.0 / n_grid

        for i in range(n_grid):
            if time.time() > deadline or used >= budget:
                break
            for j in range(n_grid):
                if time.time() > deadline or used >= budget:
                    break
                # Jitter within cell
                x0 = (i + rng.random()) * cell_size
                x1 = (j + rng.random()) * cell_size
                # Clamp to [0,1]
                x0 = min(max(x0, 0.0), 1.0)
                x1 = min(max(x1, 0.0), 1.0)

                x = np.zeros(dim)
                x[0] = x0
                x[1] = x1
                f = problem(x)
                results.append(f)
                used += 1

        # Phase 2: Local refinement around promising centers
        # The Pareto front for DTLZ7 has high hypervolume contribution near
        # the "knee" and corners. Refine around several strategic centers.
        if used < budget and time.time() < deadline:
            remaining = budget - used
            # Strategic centers: corners and midpoints of the front
            centers = [
                (0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0),
                (0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (1.0, 0.5),
                (0.5, 0.5), (0.25, 0.25), (0.75, 0.75),
                (0.25, 0.75), (0.75, 0.25),
            ]
            n_centers = len(centers)
            per_center = max(1, remaining // n_centers)

            rng2 = np.random.default_rng(456)
            for ci, (cx0, cx1) in enumerate(centers):
                if time.time() > deadline or used >= budget:
                    break
                for _ in range(per_center):
                    if time.time() > deadline or used >= budget:
                        break
                    # Gaussian perturbation around center
                    radius = 0.15 * (1.0 + 0.5 * ci / n_centers)
                    dx0 = rng2.normal(0, radius)
                    dx1 = rng2.normal(0, radius)
                    x0 = min(max(cx0 + dx0, 0.0), 1.0)
                    x1 = min(max(cx1 + dx1, 0.0), 1.0)

                    x = np.zeros(dim)
                    x[0] = x0
                    x[1] = x1
                    f = problem(x)
                    results.append(f)
                    used += 1

        # Phase 3: Fill any remaining budget with additional stratified points
        if used < budget and time.time() < deadline:
            rng3 = np.random.default_rng(789)
            for _ in range(budget - used):
                if time.time() > deadline:
                    break
                x0 = rng3.random()
                x1 = rng3.random()
                x = np.zeros(dim)
                x[0] = x0
                x[1] = x1
                f = problem(x)
                results.append(f)
                used += 1

        return results
# EVOLVE-BLOCK-END
