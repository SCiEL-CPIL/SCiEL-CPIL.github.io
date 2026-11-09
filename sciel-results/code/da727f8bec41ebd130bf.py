"""Rewritten optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with 1/5 success rule for f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget

        # Initialize at origin (often near the attractor for f6)
        x = np.zeros(dim)
        fx = func(x)
        calls_made = 1
        best_x = x.copy()
        best_f = fx

        def remaining():
            return budget - calls_made

        def time_ok():
            return time.time() < deadline

        # (1+1)-ES with 1/5 rule
        sigma = 1.0
        success_count = 0
        window = 10  # window size for 1/5 rule
        window_count = 0

        while remaining() > 0 and time_ok():
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1

            success = fy < fx
            if success:
                x = y
                fx = fy
                success_count += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                success_count -= 1

            window_count += 1

            # 1/5 rule: adjust sigma based on success rate in window
            if window_count >= window:
                success_rate = success_count / window
                if success_rate > 0.2:
                    sigma *= 1.2
                elif success_rate < 0.1:
                    sigma *= 0.8
                success_count = 0
                window_count = 0

            # Early termination if sigma is too small
            if sigma < 1e-7:
                break

        # Final polish with Nelder-Mead if budget remains
        if remaining() > 50 and time_ok():
            try:
                from scipy.optimize import minimize

                def wrapper(x):
                    if calls_made >= budget:
                        return best_f
                    val = func(x)
                    calls_made += 1
                    if val < best_f:
                        best_f = val
                        best_x = x.copy()
                    return val

                res = minimize(
                    wrapper,
                    best_x,
                    method="Nelder-Mead",
                    options={"maxiter": 100, "xatol": 1e-8, "fatol": 1e-10},
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
