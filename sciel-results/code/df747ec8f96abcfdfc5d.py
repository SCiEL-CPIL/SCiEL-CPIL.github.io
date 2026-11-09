"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Pure Multi-Start Local Search: 30 restarts, 10 evals each, fixed-step coordinate descent."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        n_restarts = 30
        evals_per_restart = budget // n_restarts  # 10
        if evals_per_restart < 1:
            evals_per_restart = 1

        best_x = None
        best_f = np.inf

        # Use Latin Hypercube Sampling for better initial spread
        # Generate 30 points in [0, 1]^dim, then scale to [-5, 5]^dim
        lhs = rng.uniform(0, 1, (n_restarts, dim))
        # Permute each column to ensure coverage
        for d in range(dim):
            perm = rng.permutation(n_restarts)
            lhs[:, d] = lhs[perm, d]
        lhs = lhs * 10.0 - 5.0

        step = 0.5

        for r in range(n_restarts):
            x = lhs[r].copy()
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            remaining = evals_per_restart - 1
            coord = 0

            while remaining > 0:
                # Cycle through coordinates
                d = coord % dim
                coord += 1

                # Try positive direction
                y = x.copy()
                y[d] = np.clip(y[d] + step, -5, 5)
                fy = func(y)
                remaining -= 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Try negative direction if we have budget
                    if remaining > 0:
                        y = x.copy()
                        y[d] = np.clip(y[d] - step, -5, 5)
                        fy = func(y)
                        remaining -= 1
                        if fy < fx:
                            x, fx = y, fy
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
