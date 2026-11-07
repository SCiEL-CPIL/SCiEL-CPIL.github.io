"""Rewritten optimizer for BBOB-style f23 (Katsuura)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 270  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Phase 1: Global exploration with random restarts
        # We'll allocate budget across a few restarts, each with its own local search
        best_x = None
        best_f = np.inf

        # Use a simple strategy: 4 restarts with 125 calls each
        n_restarts = 4
        calls_per_restart = budget // n_restarts

        call_count = 0

        for r in range(n_restarts):
            if time.time() >= deadline:
                break

            # Initialize with random point, biased toward center for later restarts
            if r == 0:
                x = rng.uniform(-5, 5, dim)
            else:
                # Gradually shrink the range
                span = 10.0 * (0.7 ** r)
                x = rng.uniform(-span / 2, span / 2, dim)

            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Local (1+1)-ES with adaptive step size
            sigma = 1.5
            success = 0
            window = 10

            for i in range(calls_per_restart - 1):
                if time.time() >= deadline:
                    break
                if call_count >= budget:
                    break

                # Generate trial point
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                call_count += 1

                if fy < fx:
                    # Success
                    x = y
                    fx = fy
                    sigma *= 1.1
                    success += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9

                # Adaptive sigma based on success rate
                if (i + 1) % window == 0:
                    rate = success / window
                    if rate > 0.3:
                        sigma *= 1.2
                    elif rate < 0.1:
                        sigma *= 0.7
                    success = 0

            # Reset x for next restart if we found something better
            if best_x is not None and r < n_restarts - 1:
                # Perturb best found so far for next restart
                x = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5)
                fx = func(x)
                call_count += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 2: Fine-tuning around the best found
        remaining = budget - call_count
        if remaining > 10 and best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1

            for i in range(remaining):
                if time.time() >= deadline:
                    break
                if call_count >= budget:
                    break

                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                call_count += 1

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.05
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.95

        return best_x, best_f
# EVOLVE-BLOCK-END
