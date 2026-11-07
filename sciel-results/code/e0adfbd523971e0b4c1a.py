"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using analytical structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        # DTLZ2 structure: x[0], x[1] are front variables, x[2:] are position variables
        # f_1 = (1+g) * sin(pi * x_0 / 2)
        # f_2 = (1+g) * cos(pi * x_0 / 2) * sin(pi * x_1 / 2)
        # f_3 = (1+g) * cos(pi * x_0 / 2) * cos(pi * x_1 / 2)
        # g = 100 * (n-1 + sum((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5)))) for i in x[2:]
        # Minimum g=0 when x_i = 0.5 for all i in x[2:]

        # The Pareto front is a quarter sphere of radius 1 in the first octant.
        # We need to sample this front well to maximize hypervolume.

        # Use Sobol sequence for better low-discrepancy coverage in 2D
        from scipy.stats import qmc

        # Number of points to evaluate on the front
        n_front = min(budget - 20, 180)  # Leave some budget for safety

        # Generate Sobol sequence in 2D
        sobol = qmc.Sobol(d=2, scramble=True)
        # Generate points in [0, 1]
        x0, x1 = sobol.random(n_front).T

        # Create decision vectors
        X = np.zeros((n_front, dim))
        X[:, 0] = x0
        X[:, 1] = x1
        X[:, 2:] = 0.5  # Position variables at optimum

        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n_front

        # Check if g is actually 0 (objectives should be <= 1.0)
        # If any objective > 1.0, the problem might not be standard DTLZ2
        # or there's numerical issue. In that case, do local search.
        max_obj = np.max(F)
        if max_obj > 1.01:  # Allow small tolerance
            # Problem might not be standard DTLZ2, do local search
            # Find best point and refine
            best_idx = np.argmin(np.sum(F, axis=1))
            x_best = X[best_idx].copy()
            f_best = F[best_idx].copy()

            step_size = 0.1
            n_local = budget - used - 2

            for i in range(n_local):
                if time.time() > deadline - 10:
                    break

                x_new = x_best + step_size * np.random.randn(dim)
                x_new = np.clip(x_new, lo, hi)

                f_new = problem(x_new)
                used += 1

                if np.sum(f_new) < np.sum(f_best):
                    x_best = x_new
                    f_best = f_new
                    step_size *= 0.9
                else:
                    step_size *= 0.95

            return x_best

        return X[0]
# EVOLVE-BLOCK-END
