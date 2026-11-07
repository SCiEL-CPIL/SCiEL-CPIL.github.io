import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """(1+1)-ES with adaptive sigma and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        calls = 1

        best_x = x.copy()
        best_fx = fx
        sigma = 3.0

        # Phase 1: (1+1)-ES with restarts for most of the budget
        # Reserve last 80 calls for Nelder-Mead polish
        polish_budget = 80
        es_budget = budget - polish_budget

        # Track progress for restart decisions
        no_improve_count = 0
        restart_threshold = 60  # restart if no improvement for this many steps
        max_sigma = 10.0
        min_sigma = 1e-12

        for _ in range(es_budget - 1):
            if time.time() >= deadline:
                break
            if calls >= budget:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.15
                no_improve_count = 0
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                sigma *= 0.92
                no_improve_count += 1

            # Clamp sigma
            if sigma > max_sigma:
                sigma = max_sigma
            elif sigma < min_sigma:
                sigma = min_sigma

            # Restart if stuck: jump to a new random point
            if no_improve_count >= restart_threshold:
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                calls += 1
                sigma = 5.0
                no_improve_count = 0
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx

        # Phase 2: Nelder-Mead polish using remaining budget
        remaining = budget - calls
        if remaining > 5 and time.time() < deadline:
            # Wrap func to track calls and enforce deadline
            nm_calls = [0]
            
            def nm_func(z):
                if time.time() >= deadline:
                    raise Exception("timeout")
                if nm_calls[0] >= remaining:
                    raise Exception("budget exceeded")
                nm_calls[0] += 1
                return func(np.clip(z, -5, 5))
            
            # Use Nelder-Mead from best point with tighter tolerances
            try:
                result = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining,
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True
                    }
                )
                if result.fun < best_fx:
                    best_x = result.x
                    best_fx = result.fun
            except Exception:
                pass

        return best_x, best_fx
