"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """LHS multi-start (1+1)-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        calls_made = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            val = func(x)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: LHS global sampling for diverse starting points
        rng = np.random.default_rng(42)
        n_starts = 30
        for i in range(n_starts):
            if not check_time() or calls_made >= budget:
                break
            # Latin Hypercube Sampling for dim=5
            x = np.empty(dim)
            for d in range(dim):
                perm = rng.permutation(n_starts)
                x[d] = -5.0 + 10.0 * (perm[i] + rng.random()) / n_starts
            x = np.clip(x, -5, 5)
            fx = safe_call(x)
            if fx is None:
                break

        if best_x is None:
            # Fallback: single random start
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is not None:
                best_f = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES refinement from best point
        if best_x is not None and check_time() and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 1.0
            success_count = 0
            total_steps = 0
            window = 10  # 1/5 success rule window

            while check_time() and calls_made < budget:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break

                total_steps += 1
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    sigma *= 1.3
                else:
                    sigma *= 0.9

                # Adaptive sigma adjustment based on success rate
                if total_steps >= window:
                    success_rate = success_count / total_steps
                    if success_rate > 0.4:
                        sigma *= 1.1
                    elif success_rate < 0.1:
                        sigma *= 0.8
                    success_count = 0
                    total_steps = 0

                # Clamp sigma to reasonable bounds
                sigma = max(sigma, 1e-6)
                sigma = min(sigma, 10.0)

                # If we've been stuck, try a small random jump
                if total_steps % 50 == 0 and sigma < 0.1:
                    x = np.clip(x + 0.5 * rng.standard_normal(dim), -5, 5)
                    fx = safe_call(x)
                    if fx is None:
                        break
                    sigma = 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
