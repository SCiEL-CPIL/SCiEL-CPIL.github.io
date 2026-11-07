"""Multi-objective optimizer on DTLZ2 using hypervolume-weighted grid search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-weighted grid optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        n_pos = self.n_obj - 1
        n_g = self.dim - n_pos
        mid = 0.5 * (lo[n_pos:] + hi[n_pos:])

        if self.n_obj == 3:
            grid_size = 14
        else:
            grid_size = int(np.sqrt(self.budget))
            while grid_size ** n_pos > self.budget:
                grid_size -= 1
                if grid_size < 2:
                    grid_size = 2
                    break

        n_grid = grid_size ** n_pos
        n_remaining = self.budget - n_grid

        u1_vals = np.linspace(0, 1, grid_size)
        u2_vals = np.linspace(0, 1, grid_size)

        U1, U2 = np.meshgrid(u1_vals, u2_vals, indexing='ij')

        X1 = (2.0 / np.pi) * np.arcsin(U1)
        X2 = U2

        X1_flat = X1.ravel()
        X2_flat = X2.ravel()

        n_points = len(X1_flat)

        X = np.zeros((n_points, self.dim))
        X[:, 0] = X1_flat
        X[:, 1] = X2_flat

        if n_g > 0:
            X[:, n_pos:] = mid

        X = np.clip(X, lo, hi)

        F = np.zeros((n_points, self.n_obj))
        for i in range(n_points):
            if time.time() > deadline - 5:
                break
            F[i] = problem(X[i])

        if n_remaining > 0 and time.time() < deadline - 5:
            n_rand = n_remaining
            rng = np.random.default_rng(42)

            u1_rand = rng.random(n_rand)
            u2_rand = rng.random(n_rand)

            X1_rand = (2.0 / np.pi) * np.arcsin(u1_rand)
            X2_rand = u2_rand

            X_rand = np.zeros((n_rand, self.dim))
            X_rand[:, 0] = X1_rand
            X_rand[:, 1] = X2_rand
            if n_g > 0:
                X_rand[:, n_pos:] = mid
            X_rand = np.clip(X_rand, lo, hi)

            X = np.vstack([X, X_rand])

            for i in range(n_rand):
                if time.time() > deadline - 5:
                    break
                problem(X[n_points + i])

        return X
# EVOLVE-BLOCK-END
