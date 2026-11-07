"""Optimized (1+1)-ES with Nelder-Mead final phase for BBOB f20 (Schwefel)."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and Nelder-Mead final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s kill

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def count_call():
            nonlocal call_count
            call_count += 1
            return call_count

        def run_es(x0, fx0, sigma0, max_calls, rng_local):
            """Run a (1+1)-ES trajectory."""
            nonlocal best_x, best_f
            x = x0.copy()
            fx = fx0
            sigma = sigma0
            calls_used = 0

            for _ in range(max_calls):
                if time.time() > deadline:
                    break
                if call_count >= budget:
                    break

                # Generate trial
                step = sigma * rng_local.standard_normal(n)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                count_call()

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15
                else:
                    sigma *= 0.9

                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            return x, fx, calls_used

        # Phase 1: Initial exploration with multiple restarts
        sigma_init = 3.0
        remaining = budget - 1  # reserve 1 for initial eval

        # First evaluation
        x0 = rng.uniform(-5, 5, n)
        fx0 = func(x0)
        count_call()
        best_x = x0.copy()
        best_f = fx0
        remaining -= 1

        # Adaptive restart strategy
        total_calls = 1

        while total_calls < budget and time.time() < deadline:
            # Determine how many calls to give this run
            frac = 0.25
            run_budget = max(10, int(remaining * frac))
            run_budget = min(run_budget, remaining)

            x, fx, used = run_es(x0, fx0, sigma_init, run_budget, rng)
            total_calls += used
            remaining -= used

            # If we made progress, continue from this point
            # If no progress, restart from a new random point
            if fx < best_f - 1e-10:
                # Made progress, continue refining from here
                x0, fx0 = x, fx
                sigma_init = max(0.1, sigma_init * 0.8)
            else:
                # No progress, restart from new random point
                x0 = rng.uniform(-5, 5, n)
                fx0 = func(x0)
                count_call()
                total_calls += 1
                remaining -= 1
                if fx0 < best_f:
                    best_f = fx0
                    best_x = x0.copy()
                sigma_init = 3.0

        # Phase 2: Nelder-Mead final polish
        if best_x is not None and time.time() < deadline - 10.0 and remaining > 10:
            # Wrap func to track calls
            nm_calls = [0]

            def nm_func(x):
                if time.time() > deadline:
                    return best_f + 1e10
                if call_count >= budget:
                    return best_f + 1e10
                x_clipped = np.clip(x, -5, 5)
                fx = func(x_clipped)
                count_call()
                nm_calls[0] += 1
                return fx

            # Run Nelder-Mead from the best point found
            try:
                nm_result = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': min(remaining, 150),
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if nm_result.fun < best_f:
                    best_f = nm_result.fun
                    best_x = np.clip(nm_result.x, -5, 5).copy()
            except Exception:
                pass

        # Phase 3: Fine-grained local search if time permits
        if best_x is not None and time.time() < deadline - 5.0 and call_count < budget - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            for _ in range(min(budget - call_count, 50)):
                if time.time() > deadline or call_count >= budget:
                    break
                step = sigma * rng.standard_normal(n)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                count_call()
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
