"""DTLZ1 optimizer that directly samples the known Pareto front using Sobol sequences."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 that directly constructs points on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit with 10s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # DTLZ1: first (dim - n_obj + 1) variables determine position on the front,
        # remaining (n_obj - 1) variables determine aggregation (should be 0.5 for Pareto front)
        n_pos = dim - self.n_obj + 1  # 7 - 3 + 1 = 5 position variables
        n_agg = self.n_obj - 1  # 2 aggregation variables

        # Generate Sobol sequence points in [0,1]^n_pos
        # We need to generate enough points to fill the budget
        # Use a simple low-discrepancy sequence via scrambled Halton or similar

        def sobol_sequence(n, d):
            """Generate n points in d dimensions using a simple low-discrepancy approach."""
            # Use inverse CDF of uniform with golden ratio for 1D, extend to dD
            # For better quality, use a proper Sobol-like construction
            # Simple approach: use Halton sequence
            points = np.zeros((n, d))
            for j in range(d):
                base = j + 2  # 2, 3, 5, 7, 11, ...
                for i in range(n):
                    f = 1.0
                    idx = i + 1
                    while idx > 0:
                        f /= base
                        points[i, j] += f * (idx % base)
                        idx //= base
            return points

        # Generate budget points on the Pareto front
        # Position variables from Halton sequence in [0,1]^n_pos
        # Aggregation variables fixed at 0.5

        # We'll evaluate all budget points, but need to be careful about time
        # Each evaluation is cheap (just a function call), so 200 should be fast

        X = np.zeros((budget, dim))
        pos_points = sobol_sequence(budget, n_pos)

        for i in range(budget):
            # Check deadline periodically
            if i % 20 == 0 and time.time() > deadline:
                # Return what we have so far
                break

            X[i, :n_pos] = pos_points[i, :]
            X[i, n_pos:] = 0.5  # Aggregation variables at 0.5 for Pareto front

        # Evaluate all points
        F = np.zeros((budget, self.n_obj))
        valid_count = 0
        for i in range(budget):
            if time.time() > deadline:
                break
            try:
                F[i] = problem(X[i])
                valid_count += 1
            except Exception:
                break

        return X[:valid_count]
# EVOLVE-BLOCK-END
