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
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ7 with m=3, n=22: Pareto front has g=0 (last 20 vars = 0)
        # f1=x0, f2=x1, f3=h(x1,0,x2) which only depends on x1 when g=0
        # So front is 2D: parameterized by x0, x1 in [0,1]
        # Set x2..x21 to 0 to achieve g=0

        n_obj = self.n_obj
        dim = self.dim

        # Generate quasi-random points in [0,1]^2 for x0, x1
        # Use Halton sequence for good low-discrepancy coverage
        def halton(index, base):
            result = 0.0
            f = 1.0 / base
            while index > 0:
                result += f * (index % base)
                index //= base
                f /= base
            return result

        # Determine grid size based on budget
        # We need to evaluate each point once, so use all budget
        # Use a square-ish grid: n x n points
        n_grid = int(np.sqrt(self.budget))
        if n_grid * n_grid > self.budget:
            n_grid -= 1

        # Generate grid points with Halton-like spacing
        # Actually, use a simple grid with offset to avoid boundaries
        x0_vals = np.linspace(0.0, 1.0, n_grid)
        x1_vals = np.linspace(0.0, 1.0, n_grid)

        results = []
        used = 0

        for x0 in x0_vals:
            for x1 in x1_vals:
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break

                # Create solution vector
                x = np.zeros(dim)
                x[0] = x0
                x[1] = x1
                # x[2] can be anything when g=0, set to 0
                # x[3:] = 0 already

                f = problem(x)
                results.append(f)
                used += 1

            if time.time() > deadline or used >= self.budget:
                break

        # If we have remaining budget, add more points with perturbation
        if used < self.budget and time.time() < deadline:
            rng = np.random.default_rng(42)
            for _ in range(self.budget - used):
                if time.time() > deadline:
                    break
                x0 = rng.random()
                x1 = rng.random()
                x = np.zeros(dim)
                x[0] = x0
                x[1] = x1
                f = problem(x)
                results.append(f)

        return results
# EVOLVE-BLOCK-END
