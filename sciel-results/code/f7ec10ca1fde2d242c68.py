"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with 1/5 success rule and final Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0
        rng = np.random.default_rng(0)

        # Initial point: center of box
        x = np.zeros(self.dim)
        fx = func(x)
        calls = 1

        sigma = 2.0
        window = 10
        successes = 0

        es_calls = 0
        max_es_calls = self.budget - 50  # reserve 50 for polish

        for i in range(max_es_calls):
            if time.time() >= deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            es_calls += 1

            if fy < fx:
                x, fx = y, fy
                successes += 1
            else:
                successes += 0

            # 1/5 success rule: check every 'window' steps
            if (i + 1) % window == 0:
                rate = successes / window
                if rate > 0.2:
                    sigma *= 1.3
                elif rate < 0.2:
                    sigma *= 0.7
                successes = 0

        # Final polish with Nelder-Mead if we have budget remaining
        remaining = self.budget - calls
        if remaining > 20 and time.time() < deadline - 5.0:
            try:
                result = minimize(
                    func, x, method='Nelder-Mead',
                    options={'maxiter': remaining, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                calls += remaining  # approximate; actual calls tracked by evaluator
                if result.fun < fx:
                    x, fx = result.x, result.fun
            except Exception:
                pass

        return x, fx
# EVOLVE-BLOCK-END
