"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= budget:
                return None
            try:
                fx = func(np.asarray(x, dtype=np.float64))
                calls_used += 1
                if np.isfinite(fx):
                    return float(fx)
                return None
            except Exception:
                return None

        def check_time():
            return time.time() >= deadline

        # Multi-restart (1+1)-ES
        # Try several different starting points and step sizes
        restarts = [
            (np.zeros(dim), 1.0),
            (np.full(dim, 1.0), 0.5),
            (np.full(dim, -1.0), 0.5),
            (np.random.default_rng(42).uniform(-3, 3, dim), 0.8),
            (np.random.default_rng(123).uniform(-3, 3, dim), 0.8),
        ]

        for init_x, init_sigma in restarts:
            if check_time() or calls_used >= budget:
                break

            x = init_x.copy()
            fx = safe_call(x)
            if fx is None:
                continue
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = init_sigma
            budget_left = budget - calls_used
            max_iter = budget_left

            for _ in range(max_iter):
                if check_time() or calls_used >= budget:
                    break

                y = np.clip(x + sigma * np.random.default_rng().standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.3
                else:
                    sigma *= 0.9

                # Early stopping if sigma is too small
                if sigma < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
