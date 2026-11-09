"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Iterated local search with adaptive coordinate-wise line search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        lb = -5.0
        ub = 5.0

        best_x = None
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_f, best_x
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def coord_search(x, f, step_sizes, max_rounds=20):
            """Coordinate-wise pattern search with adaptive steps."""
            nonlocal calls
            x = x.copy()
            for _ in range(max_rounds):
                if calls >= budget:
                    break
                improved = False
                for i in range(n):
                    if calls >= budget:
                        break
                    for direction in [1, -1]:
                        if calls >= budget:
                            break
                        x_try = x.copy()
                        x_try[i] += direction * step_sizes[i]
                        x_try = np.clip(x_try, lb, ub)
                        if np.allclose(x_try, x):
                            continue
                        f_try = evaluate(x_try)
                        if f_try < f:
                            x = x_try
                            f = f_try
                            improved = True
                            step_sizes[i] = max(step_sizes[i] * 0.7, 0.001)
                        else:
                            step_sizes[i] = min(step_sizes[i] * 1.3, 5.0)
                if not improved:
                    break
            return x, f

        # Phase 1: Global random restarts (50% budget)
        global_calls = int(budget * 0.5)
        while calls < global_calls:
            x = rng.uniform(lb, ub, size=n)
            f = evaluate(x)
            if calls >= budget:
                break
            step_sizes = np.ones(n) * 0.3
            x, f = coord_search(x, f, step_sizes, max_rounds=8)

        # Phase 2: Intensive multi-scale local search (50% budget)
        if best_x is not None and calls < budget:
            x = best_x.copy()
            f = best_f
            for scale in [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005]:
                if calls >= budget:
                    break
                step_sizes = np.ones(n) * scale
                x, f = coord_search(x, f, step_sizes, max_rounds=15)

        return best_x, best_f
# EVOLVE-BLOCK-END
