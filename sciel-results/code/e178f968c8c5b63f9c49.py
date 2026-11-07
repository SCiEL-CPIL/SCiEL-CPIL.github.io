"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with Nelder-Mead polish for Büche-Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        total_calls = 0

        def evaluate(x):
            nonlocal best_x, best_f, total_calls
            x = np.clip(x, -5, 5)
            f = func(x)
            total_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Quick random exploration to find a good starting point
        explore_budget = min(50, budget // 10)
        np.random.seed(42)
        for _ in range(explore_budget):
            if time.time() > deadline or total_calls >= budget:
                break
            x = np.random.uniform(-5, 5, n)
            evaluate(x)

        if best_x is None:
            x = np.zeros(n)
            evaluate(x)

        x = best_x.copy()
        f_x = best_f
        sigma = 1.0  # Start with moderate step size

        remaining_calls = budget - total_calls
        last_improve_call = total_calls

        # Phase 2: (1+1)-ES refinement
        while remaining_calls > 0:
            if time.time() > deadline:
                break

            # Check for stagnation: if sigma is very small and no improvement recently
            if sigma < 1e-4 and (total_calls - last_improve_call) > 30:
                # Try to escape: kick from best with larger sigma
                x = best_x + np.random.standard_normal(n) * 0.5
                x = np.clip(x, -5, 5)
                f_x = evaluate(x)
                remaining_calls -= 1
                if f_x < best_f:
                    last_improve_call = total_calls
                else:
                    # Reset to best and try with larger sigma
                    x = best_x.copy()
                    f_x = best_f
                    sigma = 0.3
                continue

            # Generate offspring
            z = np.random.standard_normal(n)
            y = x + sigma * z
            y = np.clip(y, -5, 5)

            f_y = evaluate(y)
            remaining_calls -= 1

            if f_y < f_x:
                x = y
                f_x = f_y
                sigma *= 1.05
                last_improve_call = total_calls
            else:
                sigma *= 0.95

            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)

            # If we have very few calls left, reduce sigma to ensure we don't jump out of the basin
            if remaining_calls < 20:
                sigma = min(sigma, 0.01)

        # Phase 3: Nelder-Mead polish if we have enough budget left
        if remaining_calls > 30 and best_x is not None:
            try:
                # Define a wrapper that counts calls
                nm_calls = [0]
                def nm_func(x):
                    nm_calls[0] += 1
                    if nm_calls[0] > remaining_calls - 5:
                        return best_f  # Stop if we're running out of budget
                    return evaluate(x)

                # Run Nelder-Mead with limited iterations
                result = minimize(nm_func, best_x, method='Nelder-Mead',
                                  options={'maxiter': min(remaining_calls // 2, 200),
                                           'xatol': 1e-6, 'fatol': 1e-8})

                if result.x is not None and result.fun < best_f:
                    best_x = result.x.copy()
                    best_f = result.fun

            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
