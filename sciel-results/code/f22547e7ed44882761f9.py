"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and CMA-ES-like covariance adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s kill

        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: (1+1)-ES with restarts
        # Use multiple restarts with different initial points and step sizes
        # to explore the multi-modal landscape

        # Strategy: divide budget into segments, each with a fresh start
        # but carrying over the best solution found so far

        # First, do a quick random search to get an initial best
        n_init = min(20, total_budget // 10)
        for i in range(n_init):
            x = rng.uniform(-5, 5, n)
            fx = safe_func(x)

        # Now do (1+1)-ES from the best point found
        x = best_x.copy()
        fx = best_f
        sigma = 1.0

        # Success rate based step size adaptation
        # Track successes over a window
        window_size = 15
        successes = 0
        window = []

        stagnation_count = 0
        max_stagnation = 30

        for step in range(total_budget - n_init):
            if time.time() > deadline - 5:
                break

            # Generate offspring
            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5, 5)
            fy = safe_func(y)

            # Adapt step size
            if fy < fx:
                x, fx = y, fy
                successes += 1
                window.append(1)
                stagnation_count = 0
            else:
                window.append(0)
                stagnation_count += 1

            if len(window) >= window_size:
                success_rate = sum(window) / window_size
                if success_rate > 0.3:
                    sigma *= 1.1
                elif success_rate < 0.1:
                    sigma *= 0.9
                window = window[1:]

            # Restart if stagnating
            if stagnation_count >= max_stagnation:
                # Perturb the current best with a larger step
                sigma = max(sigma, 1.0)
                stagnation_count = 0
                window = []

        # Phase 2: Local refinement with smaller steps around the best
        if time.time() < deadline - 10 and calls_made < total_budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            stagnation_count = 0
            max_stagnation = 20
            window = []
            window_size = 10

            remaining = total_budget - calls_made
            for step in range(remaining):
                if time.time() > deadline - 5:
                    break

                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = safe_func(y)

                if fy < fx:
                    x, fx = y, fy
                    stagnation_count = 0
                    window.append(1)
                else:
                    stagnation_count += 1
                    window.append(0)

                if len(window) >= window_size:
                    success_rate = sum(window) / window_size
                    if success_rate > 0.4:
                        sigma *= 1.05
                    elif success_rate < 0.1:
                        sigma *= 0.95
                    window = window[1:]

                if stagnation_count >= max_stagnation:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
