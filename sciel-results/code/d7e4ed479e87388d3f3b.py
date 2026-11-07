"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        # Helper: run a single (1+1)-ES trajectory
        def run_es(x0, fx0, sigma0, max_calls, rng_local):
            nonlocal best_x, best_f
            x = x0.copy()
            fx = fx0
            sigma = sigma0
            calls_used = 0
            for _ in range(max_calls):
                if time.time() > deadline:
                    break
                # Generate trial
                step = sigma * rng_local.standard_normal(n)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.25
                else:
                    sigma *= 0.85
                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            return x, fx, calls_used

        # Phase 1: Initial exploration with multiple restarts
        # Use larger sigma for initial jumps
        sigma_init = 3.0
        remaining = budget - 1  # reserve 1 for initial eval

        # First evaluation
        x0 = rng.uniform(-5, 5, n)
        fx0 = func(x0)
        best_x = x0.copy()
        best_f = fx0
        remaining -= 1

        # Adaptive restart strategy
        # Track progress to decide when to restart
        calls_in_current_run = 0
        total_calls = 1

        while total_calls < budget and time.time() < deadline:
            # Determine how many calls to give this run
            # Use a fraction of remaining budget, decreasing over time
            frac = 0.3
            run_budget = max(10, int(remaining * frac))
            run_budget = min(run_budget, remaining)

            x, fx, used = run_es(x0, fx0, sigma_init, run_budget, rng)
            total_calls += used
            remaining -= used

            # If we made progress, continue from this point
            # If no progress (fx >= best_f), restart from a new random point
            if fx < best_f - 1e-10:
                # Made progress, continue refining from here
                x0, fx0 = x, fx
                sigma_init = max(0.1, sigma_init * 0.7)
            else:
                # No progress, restart from new random point
                x0 = rng.uniform(-5, 5, n)
                fx0 = func(x0)
                total_calls += 1
                remaining -= 1
                if fx0 < best_f:
                    best_f = fx0
                    best_x = x0.copy()
                sigma_init = 3.0

        # Final polish: if we have time and calls left, do a fine-grained local search
        if best_x is not None and time.time() < deadline - 5.0 and remaining > 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(min(remaining, 50)):
                if time.time() > deadline:
                    break
                step = sigma * rng.standard_normal(n)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.8
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
