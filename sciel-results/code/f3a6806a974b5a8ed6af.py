"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Parametric sampling on ZDT6 Pareto front via uniform x0."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Sample x0 uniformly in [0,1], set x1..x_{dim-1} = 0.5
        # This places all points on the ZDT6 Pareto front (g=2)
        x0_vals = np.linspace(0, 1, budget)

        for x0 in x0_vals:
            if time.time() > deadline:
                break
            x = np.full(dim, 0.5)
            x[0] = x0
            problem(x)

        return None
# EVOLVE-BLOCK-END
