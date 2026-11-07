"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with 1/5 success rule and local polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        calls_made = 1
        sigma = 2.0
        start_time = time.time()
        time_limit = 280.0  # Safety margin for 300s limit
        
        # Reserve 100 calls for local optimization
        reserved_calls = 100
        es_iterations = self.budget - 1 - reserved_calls
        if es_iterations < 0:
            es_iterations = 0

        for _ in range(es_iterations):
            if time.time() - start_time > time_limit:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87

        # Local polish using L-BFGS-B
        remaining_calls = self.budget - calls_made
        if remaining_calls > 0 and time.time() - start_time < time_limit:
            def safe_func(x_in):
                nonlocal calls_made
                if calls_made >= self.budget:
                    raise StopIteration
                calls_made += 1
                return func(x_in)

            try:
                res = minimize(safe_func, x, method='L-BFGS-B',
                               bounds=[(-5.0, 5.0) for _ in range(self.dim)])
                if res.fun < fx:
                    fx = res.fun
                    x = res.x
            except StopIteration:
                pass
            except Exception:
                pass

        return x, fx
# EVOLVE-BLOCK-END
