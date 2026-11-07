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
        start_time = time.time()
        deadline = start_time + 285.0

        rng = np.random.default_rng(42)
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            x = np.asarray(x, dtype=float).flatten()
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start Powell for global exploration
        # Use 6 restarts, each with ~60 evaluations (360 total)
        n_restarts = 6
        evals_per_restart = 60

        for i in range(n_restarts):
            if time.time() > deadline - 30:
                break
            if calls_made >= budget - 50:
                break

            x0 = rng.uniform(-5, 5, dim)
            # Evaluate starting point
            fx0 = counted_func(x0)

            res = minimize(
                counted_func, x0, method='Powell',
                options={'maxfev': evals_per_restart, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()

        # Phase 2: Nelder-Mead polish around the best point
        if calls_made < budget - 10 and time.time() < deadline - 10:
            remaining = budget - calls_made
            res = minimize(
                counted_func, best_x, method='Nelder-Mead',
                options={'maxfev': remaining, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()

        # Phase 3: Coordinate descent polish for final precision
        if calls_made < budget - 5 and time.time() < deadline - 5:
            current_x = best_x.copy()
            step_sizes = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625]

            for step in step_sizes:
                if calls_made >= budget or time.time() > deadline - 3:
                    break
                improved = False
                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline - 3:
                        break
                    for direction in [-1, 1]:
                        if calls_made >= budget or time.time() > deadline - 3:
                            break
                        x_try = current_x.copy()
                        x_try[d] += direction * step
                        x_try = np.clip(x_try, -5.0, 5.0)
                        fx_try = counted_func(x_try)
                        if fx_try < best_f:
                            current_x = x_try.copy()
                            improved = True
                if not improved:
                    break

        return best_x, best_f
