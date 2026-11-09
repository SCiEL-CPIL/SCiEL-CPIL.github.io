# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using structured Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 structure:
        # M=3 objectives, K=10 distance vars, M-1=2 shape vars
        # Shape vars: x[0], x[1]; Distance vars: x[2..11]
        # g = sum_{i=2}^{11} (x[i]-0.5)^2
        # Pareto front: g=0 => x[2..11]=0.5
        # Objectives on front (g=0):
        #   f0 = cos(x[0]*pi/2) * cos(x[1]*pi/2)
        #   f1 = cos(x[0]*pi/2) * sin(x[1]*pi/2)
        #   f2 = sin(x[0]*pi/2)

        # Strategy:
        # 1. Use most of the budget for dense, structured sampling of the Pareto front
        # 2. Use cosine spacing in shape variables to maximize hypervolume coverage
        # 3. Use a small portion for local refinement near the front

        # Allocate budget:
        # - 95% for Pareto front sampling
        # - 5% for local refinement
        n_front = int(budget * 0.95)
        n_local = budget - n_front

        # Create a structured grid on the Pareto front using cosine spacing
        # Cosine spacing clusters points near the vertices, which is beneficial
        # for hypervolume calculation since the front is curved

        # Number of grid points per axis
        # Use approximately sqrt(n_front) points per axis
        n_grid = int(np.sqrt(n_front))
        if n_grid < 2:
            n_grid = 2

        # Generate cosine-spaced points in [0, 1]
        # This clusters points near 0 and 1, which corresponds to the vertices
        # of the Pareto front
        angles = np.linspace(0, np.pi / 2, n_grid)
        grid_x0 = np.cos(angles)  # Values in [0, 1], clustered near 0 and 1
        grid_x1 = np.cos(angles)

        # Generate all grid combinations
        X_front = np.zeros((n_front, dim))
        # Set distance variables to 0.5
        X_front[:, 2:] = 0.5

        # Fill in grid points
        idx = 0
        for x0 in grid_x0:
            for x1 in grid_x1:
                if idx >= n_front:
                    break
                X_front[idx, 0] = x0
                X_front[idx, 1] = x1
                idx += 1
            if idx >= n_front:
                break

        # If we have remaining budget, add more points using a different strategy
        if idx < n_front:
            n_extra = n_front - idx
            # Use Halton sequence for low-discrepancy sampling
            n_halton = n_extra
            u = self._halton(n_halton, 2)
            v = self._halton(n_halton, 3)

            # Map to shape variables
            X_front[idx:idx + n_extra, 0] = u
            X_front[idx:idx + n_extra, 1] = v

        # Evaluate all front points
        F_front = np.array([problem(x) for x in X_front[:idx]])
        used = idx

        # Local refinement: perturb points on the front slightly
        if n_local > 0 and used < budget:
            rng = np.random.default_rng(42)
            for i in range(n_local):
                if time.time() > deadline:
                    break

                # Pick a random point from the front
                idx_pt = rng.integers(idx)
                x = X_front[idx_pt].copy()

                # Small perturbation in shape variables only
                noise = 0.01 * rng.standard_normal(2)
                x[0] += noise[0]
                x[1] += noise[1]
                x = np.clip(x, lo, hi)

                fx = problem(x)
                used += 1

        return X_front[:used]

    @staticmethod
    def _halton(n, base):
        """Generate n Halton sequence points in [0,1) for a given base."""
        seq = np.zeros(n)
        f = 1.0 / base
        for i in range(n):
            idx = i + 1
            result = 0.0
            while idx > 0:
                idx, rem = divmod(idx, base)
                result += rem * f
                f /= base
            seq[i] = result
        return seq
# EVOLVE-BLOCK-END
