"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer: (1+1)-ES with 1/5 success rule for fast convergence,
    followed by a Nelder-Mead polish for fine-grained local search.
    Includes a wall-clock deadline guard to finish safely within the limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # 280s safety margin within 300s limit

        rng = np.random.default_rng(0)

        # Initial point: center of the box for f14 (Different powers)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        call_count = 1

        # (1+1)-ES with 1/5 success rule
        sigma = 2.0
        successes = 0
        total_steps = 0
        n_steps = 10  # steps per sigma adaptation

        for i in range(budget - 1):
            if time.time() > deadline:
                break
            if call_count >= budget:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            call_count += 1
            total_steps += 1

            if fy < fx:
                x, fx = y, fy
                successes += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = y.copy()
            else:
                pass

            if total_steps % n_steps == 0:
                rate = successes / n_steps
                if rate > 0.2:
                    sigma *= 1.3
                elif rate < 0.2:
                    sigma *= 0.7
                successes = 0

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 1e-6, 50.0)

        # Final polish with Nelder-Mead if time and budget permits
        remaining_budget = budget - call_count
        if time.time() < deadline - 5.0 and remaining_budget > 0:
            try:
                from scipy.optimize import minimize

                # Use Nelder-Mead starting from best_x
                maxfev = min(remaining_budget, 200)
                res = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': maxfev,
                        'maxfev': maxfev,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                    },
                )
                if res.fun < best_fx:
                    best_x = res.x
                    best_fx = res.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
