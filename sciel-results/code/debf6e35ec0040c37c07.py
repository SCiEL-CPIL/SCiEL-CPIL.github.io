"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with time guard, smart initialization, and adaptive sigma."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 30s safety margin before 320s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: Quick exploration - evaluate a few candidate points
        # Start with origin (center of box) and a few random points
        candidates = [np.zeros(dim)]
        for _ in range(3):
            candidates.append(rng.uniform(-4, 4, dim))
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        for c in candidates:
            c_clipped = np.clip(c, -5, 5)
            f_val = func(c_clipped)
            calls_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = c_clipped

        x = best_x
        fx = best_f

        # Phase 2: (1+1)-ES with adaptive sigma
        sigma = 1.5
        n_success = 0
        n_total = 0
        window_size = 10

        for i in range(budget - calls_used):
            if time.time() >= deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                x, fx = y, fy
                n_success += 1
                sigma *= 1.3  # more aggressive increase on success
            else:
                sigma *= 0.8  # more aggressive decrease on failure

            n_total += 1
            if n_total >= window_size:
                # 1/5 rule adaptation
                success_rate = n_success / n_total
                if success_rate > 0.2:
                    sigma *= 1.15
                elif success_rate < 0.2:
                    sigma *= 0.85
                n_success = 0
                n_total = 0

            # Ensure sigma stays in reasonable bounds
            sigma = np.clip(sigma, 0.01, 5.0)

        return x, fx
# EVOLVE-BLOCK-END
