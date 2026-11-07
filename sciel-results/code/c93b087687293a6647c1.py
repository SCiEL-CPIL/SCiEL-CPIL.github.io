"""Optimized Weierstrass (f16) black-box optimizer."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-like (1+1)-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s wall-clock

        # Multi-start: evaluate center and a few random points
        candidates = [np.zeros(dim)]
        rng = np.random.default_rng(42)
        for _ in range(4):
            candidates.append(rng.uniform(-5, 5, dim))

        best_x = None
        best_f = np.inf
        for c in candidates:
            fc = func(np.clip(c, -5, 5))
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        x = best_x
        fx = best_f
        sigma = 1.0
        calls = len(candidates)

        # Phase 1: (1+1)-ES with adaptive step size
        success_count = 0
        window = 10
        main_budget = int(budget * 0.7)  # Use 70% of budget for main search

        for i in range(main_budget):
            if time.time() >= deadline:
                break
            if calls >= budget:
                break

            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                success_count += 1

            # Adapt sigma every 'window' steps
            if (i + 1) % window == 0:
                rate = success_count / window
                if rate > 0.5:
                    sigma *= 1.15
                elif rate < 0.2:
                    sigma *= 0.85
                else:
                    sigma *= 1.0
                success_count = 0

        # Phase 2: High-precision polish with smaller steps
        if calls < budget and time.time() < deadline:
            sigma_polish = max(sigma * 0.1, 1e-4)
            polish_calls = budget - calls
            polish_success = 0
            polish_window = 5

            for j in range(polish_calls):
                if time.time() >= deadline:
                    break

                # Generate finer-grained offspring
                y = np.clip(x + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    x, fx = y, fy
                    polish_success += 1

                # Adapt polish sigma more aggressively
                if (j + 1) % polish_window == 0:
                    rate = polish_success / polish_window
                    if rate > 0.5:
                        sigma_polish *= 1.1
                    elif rate < 0.1:
                        sigma_polish *= 0.7
                    else:
                        sigma_polish *= 0.95
                    polish_success = 0

        return x, fx
# EVOLVE-BLOCK-END
