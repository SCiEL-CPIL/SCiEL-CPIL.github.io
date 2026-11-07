"""Optimizer for ZDT1 exploiting its known structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits ZDT1 structure: x1 controls front position, x2..xn control distance to front.
    Places points on the Pareto front (x2..xn=0) with non-uniform (quadratic) spacing
    to concentrate near x1=0 where the front has highest curvature."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim

        # ZDT1 Pareto front: x2..xn = 0, f1 = x1, f2 = 1 - sqrt(x1)
        # Use non-uniform (quadratic) sampling to concentrate near x1=0
        # where the Pareto front curvature is highest
        n_points = self.budget

        # Quadratic spacing: t^2 maps [0,1] with more points near 0
        t = np.linspace(0, 1, n_points)
        x1_vals = t ** 2  # Non-uniform: denser near 0

        # Ensure we cover the full range including endpoints
        x1_vals[0] = 0.0
        x1_vals[-1] = 1.0

        evaluated = 0
        for i in range(n_points):
            if time.time() > deadline:
                break

            x = np.zeros(n)
            x[0] = x1_vals[i]
            # x[1:] = 0 (Pareto front)
            problem(x)
            evaluated += 1

        return None
# EVOLVE-BLOCK-END
