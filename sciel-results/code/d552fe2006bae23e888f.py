import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin
        rng = np.random.default_rng(42)
        eval_count = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def wrapped(x):
            nonlocal eval_count, best_f, best_x
            if time.time() >= deadline:
                raise TimeoutError("Time limit exceeded")
            if eval_count >= budget:
                raise ValueError("Budget exceeded")
            eval_count += 1
            x = np.asarray(x, dtype=np.float64).flatten()
            x = np.clip(x, -5, 5)
            f_val = float(func(x))
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Generate and evaluate initial points
        # Use a mix of origin and random points
        starts = [np.zeros(dim)]
        for _ in range(4):
            starts.append(rng.uniform(-5, 5, dim))
        
        # Evaluate all starts
        for x0 in starts:
            if eval_count >= budget or time.time() >= deadline:
                break
            try:
                wrapped(x0)
            except (TimeoutError, ValueError):
                break

        # Phase 2: L-BFGS-B optimization from the best start
        # L-BFGS-B is efficient for Rosenbrock if started reasonably
        if eval_count < budget and time.time() < deadline:
            remaining = budget - eval_count
            # Allocate most of the budget to the first local search
            maxfev = min(remaining, 350)
            if maxfev > 10:
                try:
                    res = minimize(
                        wrapped,
                        best_x,
                        method='L-BFGS-B',
                        options={
                            'maxiter': 100,
                            'maxfev': maxfev,
                            'ftol': 1e-15,
                            'gtol': 1e-15,
                            'eps': 1e-5
                        }
                    )
                except (TimeoutError, ValueError):
                    pass
                except Exception:
                    pass

        # Phase 3: Polish with tighter tolerance if budget remains
        if eval_count < budget and time.time() < deadline:
            remaining = budget - eval_count
            if remaining > 10:
                try:
                    res = minimize(
                        wrapped,
                        best_x,
                        method='L-BFGS-B',
                        options={
                            'maxiter': 50,
                            'maxfev': remaining,
                            'ftol': 1e-16,
                            'gtol': 1e-16,
                            'eps': 1e-6
                        }
                    )
                except (TimeoutError, ValueError):
                    pass
                except Exception:
                    pass

        # Fallback: if something went wrong and no calls were made, do random search
        if eval_count == 0:
            for _ in range(min(10, budget)):
                x = rng.uniform(-5, 5, dim)
                try:
                    f_val = float(func(x))
                    eval_count += 1
                    if f_val < best_f:
                        best_f = f_val
                        best_x = x.copy()
                except Exception:
                    break

        return best_x, best_f
