# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio search: many diverse random restarts with local (1+1)-ES refinement,
       plus a global CMA-ES-like covariance adaptation on the best-so-far neighborhood.
       Uses a strict wall-clock deadline guard to stay within 300s."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self.total_budget = budget

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin
        dim = self.dim
        budget = self.total_budget

        # Global state
        best_x = None
        best_f = float('inf')
        calls_made = 0

        def clip(x):
            return np.clip(x, -5, 5)

        def remaining_time():
            return deadline - time.time()

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = clip(np.asarray(x, dtype=float))
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Strategy: Multi-start (1+1)-ES with adaptive step size
        # Phase 1: Quick broad exploration to find a good starting region
        x = rng.uniform(-5, 5, dim)
        fx = evaluate(x)

        sigma_explore = 2.0
        n_explore = min(50, budget // 5)
        for i in range(n_explore):
            if calls_made >= budget or remaining_time() < 5.0:
                break
            # Mix: 40% pure random, 60% perturbation of best
            if rng.random() < 0.4:
                y = clip(rng.uniform(-5, 5, dim))
            else:
                y = clip(best_x + sigma_explore * rng.standard_normal(dim))

            fy = evaluate(y)
            if fy < best_f - 1.0:
                sigma_explore = min(3.0, sigma_explore * 1.1)
            elif i > 0 and i % 10 == 0:
                sigma_explore = max(0.5, sigma_explore * 0.95)

        # Phase 2: Multi-start (1+1)-ES with multiple restarts
        # Each restart: local (1+1)-ES from a different starting point
        n_restarts = 0
        max_restarts = 8
        calls_per_restart = max(20, (budget - calls_made) // max_restarts)

        for restart in range(max_restarts):
            if calls_made >= budget or remaining_time() < 5.0:
                break

            # Choose starting point for this restart
            if restart == 0:
                x = best_x.copy()
            else:
                # Start from a perturbation of best, or a new random point
                if rng.random() < 0.3:
                    x = clip(rng.uniform(-5, 5, dim))
                else:
                    x = clip(best_x + 1.5 * rng.standard_normal(dim))

            fx = evaluate(x)
            sigma = 0.3
            success_count = 0
            total_count = 0

            # (1+1)-ES with 1/5 success rule
            for i in range(calls_per_restart):
                if calls_made >= budget or remaining_time() < 3.0:
                    break

                y = clip(x + sigma * rng.standard_normal(dim))
                fy = evaluate(y)
                total_count += 1

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    sigma *= 1.05
                    if sigma > 1.5:
                        sigma = 1.5
                else:
                    sigma *= 0.95
                    if sigma < 0.005:
                        sigma = 0.005

                # 1/5 success rule: adapt sigma based on success rate over last 10 steps
                if total_count % 10 == 0 and total_count >= 10:
                    success_rate = success_count / total_count
                    if success_rate > 0.3:
                        sigma *= 1.2
                    elif success_rate < 0.1:
                        sigma *= 0.8
                    success_count = 0
                    total_count = 0

            # Update global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            n_restarts += 1

        # Phase 3: Final high-precision polish around the best point
        if calls_made < budget and remaining_time() > 5.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            n_fine = min(50, budget - calls_made)
            for i in range(n_fine):
                if calls_made >= budget or remaining_time() < 2.0:
                    break
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.02
                else:
                    sigma *= 0.92
                    if sigma < 0.001:
                        sigma = 0.001

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
