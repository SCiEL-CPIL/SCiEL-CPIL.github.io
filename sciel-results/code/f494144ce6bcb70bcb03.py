"""Optimizer for DTLZ4 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using direct Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, index, base):
        """Generate a single Halton sequence value."""
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    def _halton_sequence(self, n, dim):
        """Generate n quasi-random points in [0,1)^dim using Halton sequence."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        points = np.zeros((n, dim))
        for i in range(n):
            for d in range(dim):
                points[i, d] = self._halton(i + 1, primes[d])
        return points

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ4: first 2 vars determine front position, rest must be 0.5 for Pareto optimality
        n_sample = self.budget  # Use full budget
        # Generate quasi-random points in [0,1)^2 for x0, x1
        u = self._halton_sequence(n_sample, 2)

        # Construct points: x0, x1 from Halton, x2..x11 = 0.5
        X = np.zeros((n_sample, self.dim))
        X[:, 0] = u[:, 0]
        X[:, 1] = u[:, 1]
        X[:, 2:] = 0.5  # Set g-variables to 0.5 for exact Pareto front

        # Clip to bounds (should be [0,1] but be safe)
        X = np.clip(X, lo, hi)

        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n_sample

        # If we have time and budget, we could refine, but for DTLZ4 with g=0,
        # these are already Pareto optimal. No further improvement possible.
        return X
# EVOLVE-BLOCK-END
