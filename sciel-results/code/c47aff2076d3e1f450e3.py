"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Sobol sequence-based deterministic global optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        # Generate Sobol sequence for 500 points in [0,1]^5
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        n_points = self.budget
        sobol_samples = sampler.random(n_points)

        # Map from [0,1]^dim to [-5,5]^dim
        x_coords = -5 + 10 * sobol_samples

        best_x = None
        best_f = float('inf')

        for i in range(n_points):
            # Check time deadline
            if time.time() > deadline:
                break

            x = x_coords[i]
            fx = func(x)

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback if no evaluations were made
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
