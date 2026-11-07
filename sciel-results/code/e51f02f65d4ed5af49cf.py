"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with windowed 1/5 success rule for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        global_best_x = None
        global_best_f = np.inf
        calls_made = 0

        def check_deadline():
            return time.time() > deadline

        def run_es(x_start, fx_start, max_calls, sigma_init=2.0, window=15):
            """Run (1+1)-ES with windowed 1/5 success rule for up to max_calls evaluations."""
            nonlocal calls_made, global_best_x, global_best_f

            x = x_start.copy()
            fx = fx_start
            sigma = sigma_init
            min_sigma = 1e-8
            max_sigma = 10.0
            successes = 0
            local_best_x = x.copy()
            local_best_f = fx

            for _ in range(max_calls):
                if calls_made >= budget:
                    break
                if check_deadline():
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < local_best_f:
                    local_best_f = fy
                    local_best_x = y.copy()

                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1

                # Windowed 1/5 success rule
                if calls_made % window == 0:
                    success_rate = successes / window
                    if success_rate > 0.2:
                        sigma *= 1.3
                    elif success_rate < 0.2:
                        sigma *= 0.7
                    sigma = np.clip(sigma, min_sigma, max_sigma)
                    successes = 0

            # Update global best
            if local_best_f < global_best_f:
                global_best_f = local_best_f
                global_best_x = local_best_x.copy()

            return local_best_x, local_best_f

        # Phase 1: Multi-start exploration with diverse initial points
        # Start at origin (likely near optimum due to BBOB shift constraints)
        x0 = np.zeros(dim)
        fx0 = func(x0)
        calls_made += 1

        if fx0 < global_best_f:
            global_best_f = fx0
            global_best_x = x0.copy()

        # Allocate budget: ~40% to initial exploration with multiple restarts,
        # ~60% to refinement of the best found so far
        n_restarts = 6
        budget_per_restart = max(10, (budget - 1) // (n_restarts + 2))

        # Run first ES from origin
        remaining = budget - calls_made
        if remaining > 0:
            run_es(x0, fx0, min(remaining, budget_per_restart), sigma_init=2.0)

        # Additional restarts from random points to escape local minima
        for restart in range(n_restarts - 1):
            if calls_made >= budget:
                break
            if check_deadline():
                break

            # Random start, biased toward center
            x_start = rng.uniform(-3, 3, dim)
            fx_start = func(x_start)
            calls_made += 1

            if fx_start < global_best_f:
                global_best_f = fx_start
                global_best_x = x_start.copy()

            remaining = budget - calls_made
            if remaining <= 0:
                break

            # Use smaller budget for later restarts (exploit more)
            alloc = min(remaining, budget_per_restart)
            run_es(x_start, fx_start, alloc, sigma_init=1.5)

        # Phase 2: Refinement around global best with fine-tuned step size
        if global_best_x is not None and calls_made < budget and not check_deadline():
            remaining = budget - calls_made
            # Use a smaller window for finer adaptation during refinement
            run_es(global_best_x, global_best_f, remaining, sigma_init=0.5, window=10)

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
