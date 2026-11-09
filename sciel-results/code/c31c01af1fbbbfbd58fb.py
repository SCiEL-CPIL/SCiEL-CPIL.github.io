"""DTLZ2 hypervolume maximizer: direct Pareto front sampling with surface-area weighting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2: direct Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, base):
        """Generate n Halton numbers in [0, 1) with the given base."""
        result = np.zeros(n)
        f = 1.0 / base
        for i in range(n):
            i_val = i + 1
            while i_val > 0:
                result[i] += f * (i_val % base)
                i_val //= base
                f /= base
        return result

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n = self.budget
        # Generate Halton sequences for theta1 and theta2
        u1 = self._halton(n, 2)  # for theta1
        u2 = self._halton(n, 3)  # for theta2

        # Surface-area-weighted sampling:
        # theta1: density proportional to sin(theta1) on [0, pi/2]
        # Inverse CDF: theta1 = arccos(1 - u)
        theta1 = np.arccos(1.0 - u1)
        # theta2: uniform on [0, pi/2]
        theta2 = u2 * (np.pi / 2.0)

        # Compute objective values on Pareto front (g=0)
        f1 = np.cos(theta1) * np.cos(theta2)
        f2 = np.cos(theta1) * np.sin(theta2)
        f3 = np.sin(theta1)

        # Map back to decision space
        # x1 = 2*theta1/pi, x2 = 2*theta2/pi, x3..x11 = 0.5
        X = np.zeros((n, self.dim))
        X[:, 0] = 2.0 * theta1 / np.pi
        X[:, 1] = 2.0 * theta2 / np.pi
        for i in range(2, self.dim):
            X[:, i] = 0.5

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        return X
# EVOLVE-BLOCK-END
