"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import dual_annealing


class Optimizer:
    """Dual Annealing optimizer for BBOB f1 (Sphere) with hidden shift/rotation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        # Track best solution found so far
        best_x = np.zeros(self.dim)
        best_f = np.inf

        # We'll use a counter to limit function calls
        call_count = 0

        def bounded_func(x):
            nonlocal best_x, best_f, call_count, deadline
            if time.time() > deadline:
                return best_f  # Return best known to stop further exploration
            # Clip to bounds
            x_clipped = np.clip(x, -5.0, 5.0)
            val = float(func(x_clipped))
            call_count += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Define bounds for scipy
        bounds = [(-5.0, 5.0)] * self.dim

        # Use dual_annealing which combines simulated annealing with Nelder-Mead
        # maxiter is the max number of iterations per method
        # maxfev is the max number of function evaluations
        # We set maxfev to budget to respect the call limit
        # But we also need to be careful about time

        try:
            result = dual_annealing(
                bounded_func,
                bounds=bounds,
                maxiter=100,
                maxfev=self.budget,
                initial_temp=5230.0,
                restart_temp_ratio=2e5,
                visit=2.62,
                seed=42,
                x0=None,
            )
        except Exception:
            pass

        # If we haven't used all our budget and still have time, do a local polish
        remaining_budget = self.budget - call_count
        if remaining_budget > 10 and time.time() < deadline - 5.0:
            # Use Nelder-Mead for final polish
            try:
                from scipy.optimize import minimize

                def polish_func(x):
                    nonlocal best_x, best_f, call_count, deadline
                    if time.time() > deadline:
                        return best_f
                    x_clipped = np.clip(x, -5.0, 5.0)
                    val = float(func(x_clipped))
                    call_count += 1
                    if val < best_f:
                        best_f = val
                        best_x = x_clipped.copy()
                    return val

                result_nm = minimize(
                    polish_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500,
                        'maxfev': remaining_budget,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True,
                    }
                )
                if result_nm.fun < best_f:
                    best_f = result_nm.fun
                    best_x = np.clip(result_nm.x, -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
