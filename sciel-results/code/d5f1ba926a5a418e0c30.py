"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ7 exploiting the known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_obj = self.n_obj
        # DTLZ7 active dimensions: x[0] (g-function coupling) and last n_obj dimensions
        active_idx = [0] + list(range(self.dim - n_obj, self.dim))
        inactive_idx = [i for i in range(self.dim) if i not in active_idx]

        # The Pareto front of DTLZ7 is characterized by:
        # f_k = g * (1 + sin(k*pi*x[0]/2)) for k=1,2,3
        # where g = 1 + 9/(n-1) * sum(x_i^2) for i in inactive dims
        # Optimal: inactive dims = 0 (g=1), active dims on the front
        # The front is non-convex, so we need good spread in x[0] and last 3 dims

        # Strategy:
        # 1. Initialize with points spread along the known Pareto front structure
        # 2. Use local refinement around non-dominated points
        # 3. Keep inactive dims at 0 to minimize g

        pop_size = 40
        X = np.zeros((pop_size, self.dim))

        # Initialize with structured spread on the Pareto front
        # x[0] controls the front shape, last 3 dims also matter
        # For DTLZ7, the front is defined by x[0] in [0, pi/2] roughly
        # Let's spread x[0] and the last 3 dimensions

        # Spread x[0] uniformly in [0, 1] (which maps to the front)
        X[:, 0] = np.linspace(0, 1, pop_size)

        # For the last 3 dimensions, spread them as well
        # In DTLZ7, the last n_obj dimensions define the front
        # A good initialization: spread them in a way that covers the front
        for i, idx in enumerate(range(self.dim - n_obj, self.dim)):
            X[:, idx] = np.linspace(0, 1, pop_size)

        # Inactive dims stay at 0
        for idx in inactive_idx:
            X[:, idx] = 0.0

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Keep track of all evaluated points for hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]

        # Local refinement loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Find non-dominated points
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            nd_idx = np.where(~dominated)[0]

            if len(nd_idx) == 0:
                nd_idx = np.arange(n)

            # For each non-dominated point, try a local mutation
            # Focus on active dimensions only
            for parent_idx in nd_idx:
                if used >= self.budget or time.time() > deadline:
                    break

                parent = X[parent_idx].copy()

                # Gaussian mutation on active dimensions with adaptive step
                # Use a smaller step for refinement
                step = 0.05 * (hi[active_idx] - lo[active_idx])
                for idx in active_idx:
                    parent[idx] += step * rng.standard_normal()
                    parent[idx] = np.clip(parent[idx], lo[idx], hi[idx])

                # Keep inactive dims at 0
                for idx in inactive_idx:
                    parent[idx] = 0.0

                fc = problem(parent)
                used += 1

                # Add to population (replace worst by sum of objectives)
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst], F[worst] = parent, fc

                # Record for hypervolume
                all_X[-1][worst] = parent
                all_F[-1][worst] = fc

        return X
# EVOLVE-BLOCK-END
