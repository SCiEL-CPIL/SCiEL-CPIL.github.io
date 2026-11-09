"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid global-local optimizer for rotated Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        deadline = time.time() + 280  # safety margin before 300s

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            try:
                val = func(np.asarray(x, dtype=float).reshape(dim))
                if np.isfinite(val):
                    return float(val)
                return 1e10
            except Exception:
                return 1e10

        # Phase 1: Global exploration with multiple local searches
        # Use ~60% of budget for exploration
        explore_budget = int(budget * 0.6)
        n_restarts = 8
        per_restart = explore_budget // n_restarts

        for r in range(n_restarts):
            if time.time() > deadline or call_count >= budget:
                break

            # Diverse starting points
            if r == 0:
                x = np.zeros(dim)
            elif r == 1:
                x = rng.uniform(-4, 4, dim)
            elif r == 2:
                x = rng.uniform(-3, 3, dim)
            else:
                x = rng.uniform(-5, 5, dim)

            x = np.clip(x, -5, 5)
            fx = safe_func(x)

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Local (1+1)-ES with adaptive sigma
            sigma = 1.5
            for _ in range(per_restart):
                if time.time() > deadline or call_count >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.3, 4.0)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.75

        # Phase 2: Intensive exploitation around best point
        if best_x is not None and call_count < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.8
            stagnation = 0
            max_stagnation = 30

            while call_count < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.15, 3.0)
                    stagnation = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    stagnation += 1

                if stagnation > max_stagnation:
                    # Try a few random perturbations to escape
                    for _ in range(5):
                        if call_count >= budget or time.time() > deadline:
                            break
                        y = np.clip(x + 2.0 * rng.standard_normal(dim), -5, 5)
                        fy = safe_func(y)
                        if fy < fx:
                            x, fx = y, fy
                            sigma = 1.5
                            stagnation = 0
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            break
                    else:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
