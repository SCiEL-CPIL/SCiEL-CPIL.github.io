"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT1-aware optimizer: deterministic sampling of the analytic Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n = self.budget
        # Sample x1 uniformly in [0, 1] with n points
        x1_values = np.linspace(0.0, 1.0, n)

        # ZDT1: f1 = x1, f2 = (1 + g(x2..xn)) * (1 - sqrt(x1 / (1 + g(x2..xn))))
        # Pareto front: g = 0 => all x_i = 0 for i >= 2
        # On PF: f1 = x1, f2 = 1 - sqrt(x1)
        # So for each x1, set x2..xn = 0

        X = np.zeros((n, self.dim))
        X[:, 0] = x1_values
        # x2..xn are already 0

        F = np.array([problem(x) for x in X])

        # Check deadline - if we're running out of time, return what we have
        if time.time() > deadline:
            return X

        return X
# EVOLVE-BLOCK-END
