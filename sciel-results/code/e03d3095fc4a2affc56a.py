"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """(1+1)-ES with adaptive step size and time guard for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 330s limit
        rng = np.random.default_rng(0)

        dim = self.dim
        budget = self.budget

        # Random initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        calls_made = 1

        best_x = x.copy()
        best_f = fx

        # 1/5 success rule parameters
        sigma = 2.0
        successes = 0
        window = 10

        for i in range(budget - 1):
            if time.time() >= deadline:
                break

            # Generate perturbation
            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            calls_made += 1

            # Track best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()

            # Accept/reject with 1/5 success rule
            if fy < fx:
                x = y
                fx = fy
                successes += 1

            # Update sigma every `window` steps
            if (i + 1) % window == 0:
                rate = successes / window
                sigma *= math.exp(0.33 * (rate - 0.2))
                sigma = np.clip(sigma, 1e-8, 10.0)
                successes = 0

        # Final polish: Nelder-Mead on best point if budget remains
        remaining = budget - calls_made
        if remaining >= 10 and time.time() < deadline - 5.0:
            from scipy.optimize import minimize
            try:
                res = minimize(
                    func, best_x, method='Nelder-Mead',
                    options={'maxiter': min(remaining, 200),
                             'fatol': 1e-12, 'xatol': 1e-10,
                             'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
