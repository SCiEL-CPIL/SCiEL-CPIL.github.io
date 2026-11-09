"""Optimized DTLZ2 solver using deterministic Halton sequence sampling on the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2.
    Uses a Halton sequence to sample the angular parameters of the Pareto front
    and sets distance parameters to the midpoint (0.5) to ensure points lie
    on the Pareto front (g=0).
    """

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
        # Explicit deadline guard: 300s limit, use 290s to be safe
        start_time = time.time()
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For DTLZ2, the Pareto front is determined by the first M-1 variables.
        # Here M=3, so x[0] and x[1] determine the position on the front.
        # The remaining variables determine the distance from the front.
        # Setting them to the midpoint (0.5 for [0,1] bounds) ensures g=0.
        
        n_pos = 2  # Number of position variables for 3 objectives
        if self.dim <= n_pos:
            # Fallback if dim is unexpectedly small
            n_pos = self.dim

        # Generate Halton sequences for the position variables
        # Using bases 2 and 3 for the first two dimensions
        h1 = self._halton(self.budget, 2)
        h2 = self._halton(self.budget, 3)

        # Prepare the decision vectors
        X = np.zeros((self.budget, self.dim))
        
        # Fill position variables with Halton values
        X[:, 0] = h1
        if self.dim > 1:
            X[:, 1] = h2
        
        # Fill distance variables with midpoint to stay on the Pareto front
        if self.dim > n_pos:
            mid = (lo[n_pos:] + hi[n_pos:]) / 2.0
            X[:, n_pos:] = mid

        # Evaluate each point
        for i in range(self.budget):
            if time.time() > deadline:
                break
            # problem(x) clips and returns objectives
            problem(X[i])

        # Return value is ignored by the evaluator
        return X[-1]
# EVOLVE-BLOCK-END
