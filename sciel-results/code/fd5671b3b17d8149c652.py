"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Pure Parallel Multi-Start with Fixed-Step Steepest Descent."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        n_restarts = 30
        evals_per_restart = budget // n_restarts  # 10

        best_x = None
        best_f = np.inf

        # Fixed step size for steepest descent
        step = 1.5

        for r in range(n_restarts):
            # 1. Random Initialization (1 eval)
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            remaining = evals_per_restart - 1  # 9 evals left

            # Steepest descent: each round, try all dimensions with +step,
            # pick the best improvement, move there.
            # Round 1: 5 evals (scan all dims)
            # Round 2: up to 4 evals (scan remaining dims)
            for round_idx in range(2):
                if remaining <= 0:
                    break
                # Determine which dimensions to scan this round
                if round_idx == 0:
                    dims_to_try = list(range(dim))
                else:
                    # Second round: try dimensions in reverse order for diversity
                    dims_to_try = list(range(dim - 1, -1, -1))

                # Limit by remaining budget
                dims_to_try = dims_to_try[:remaining]

                best_dim = -1
                best_fy = fx
                best_y = None

                for d in dims_to_try:
                    if remaining <= 0:
                        break
                    y = x.copy()
                    y[d] = np.clip(y[d] + step, -5, 5)
                    fy = func(y)
                    remaining -= 1
                    if fy < best_fy:
                        best_fy = fy
                        best_dim = d
                        best_y = y.copy()

                # If we found an improvement, move there
                if best_dim >= 0:
                    x = best_y
                    fx = best_fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

                # Also try negative step on the best dimension if we have budget
                if remaining > 0 and best_dim >= 0:
                    y = x.copy()
                    y[best_dim] = np.clip(y[best_dim] + step, -5, 5)
                    fy = func(y)
                    remaining -= 1
                    if fy < fx:
                        x = y
                        fx = fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
