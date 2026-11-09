"""Multi-scale coordinate descent + CMA-ES hybrid optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-scale coordinate descent for early progress + CMA-ES for refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        x_best = None
        f_best = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal x_best, f_best, calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        def run_coordinate_descent(x0, max_calls):
            """Multi-scale coordinate descent with geometrically shrinking steps."""
            nonlocal calls_made, x_best, f_best
            if calls_made >= budget or time.time() > deadline:
                return

            x = np.array(x0, dtype=np.float64)
            f = evaluate(x)
            calls_made += 1

            # Start with large steps, shrink geometrically
            step_sizes = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
            # We can cycle through these multiple times if budget allows

            total_steps_available = max_calls - 1
            if total_steps_available <= 0:
                return

            # Determine how many full passes we can afford
            # Each pass over all dims with all step sizes costs dim * len(step_sizes) calls
            # But we'll be smarter: for each step size, we do one pass over all dims

            for step in step_sizes:
                if calls_made >= budget or time.time() > deadline:
                    break

                # For this step size, try moving in each dimension
                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline:
                        break

                    # Try positive direction
                    trial = x.copy()
                    trial[d] += step
                    trial = np.clip(trial, -5, 5)
                    f_trial = evaluate(trial)

                    if f_trial < f:
                        x = trial.copy()
                        f = f_trial
                    else:
                        # Try negative direction
                        trial = x.copy()
                        trial[d] -= step
                        trial = np.clip(trial, -5, 5)
                        f_trial = evaluate(trial)
                        if f_trial < f:
                            x = trial.copy()
                            f = f_trial

            # Continue with smaller steps if budget remains
            for step in [0.004, 0.002, 0.001, 0.0005, 0.00025, 0.000125]:
                if calls_made >= budget or time.time() > deadline:
                    break

                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline:
                        break

                    trial = x.copy()
                    trial[d] += step
                    trial = np.clip(trial, -5, 5)
                    f_trial = evaluate(trial)

                    if f_trial < f:
                        x = trial.copy()
                        f = f_trial
                    else:
                        trial = x.copy()
                        trial[d] -= step
                        trial = np.clip(trial, -5, 5)
                        f_trial = evaluate(trial)
                        if f_trial < f:
                            x = trial.copy()
                            f = f_trial

        def run_cmaes(x0, max_calls):
            """Run CMA-ES style optimization using scipy's minimize with Nelder-Mead as fallback."""
            nonlocal calls_made, x_best, f_best
            if x0 is None or max_calls < 5 or time.time() > deadline:
                return

            local_x = np.array(x0, dtype=np.float64)
            local_f = f_best

            def wrapped(x):
                nonlocal calls_made
                if calls_made >= budget or time.time() > deadline:
                    return local_f
                x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
                fx = float(func(x))
                calls_made += 1
                if fx < local_f:
                    local_f = fx
                    x_best = x.copy()
                return fx

            try:
                # Use Nelder-Mead with tight tolerances for final polish
                res = minimize(wrapped, local_x, method='Nelder-Mead',
                               options={'maxiter': max_calls, 'maxfev': max_calls,
                                        'xatol': 1e-8, 'fatol': 1e-10,
                                        'adaptive': True})
                if res.fun < f_best:
                    f_best = float(res.fun)
                    x_best = np.array(res.x, dtype=np.float64)
            except Exception:
                pass

        # Phase 1: Multi-scale coordinate descent from origin for rapid early progress
        cd_calls = int(budget * 0.35)
        if time.time() < deadline:
            run_coordinate_descent(np.zeros(dim), cd_calls)

        # Phase 2: Nelder-Mead refinement on best point
        remaining = budget - calls_made
        if remaining >= 10 and x_best is not None and time.time() < deadline:
            nm_calls = min(remaining, 30)
            run_cmaes(x_best.copy(), nm_calls)

        # Phase 3: Second coordinate descent from a different starting point
        remaining = budget - calls_made
        if remaining >= 20 and time.time() < deadline:
            rng = np.random.default_rng(42)
            x_start2 = rng.uniform(-3, 3, dim)
            cd_calls2 = min(remaining, int(remaining * 0.4))
            run_coordinate_descent(x_start2, cd_calls2)

        # Phase 4: Another Nelder-Mead refinement
        remaining = budget - calls_made
        if remaining >= 10 and x_best is not None and time.time() < deadline:
            nm_calls2 = min(remaining, 25)
            run_cmaes(x_best.copy(), nm_calls2)

        # Phase 5: Third coordinate descent if budget remains
        remaining = budget - calls_made
        if remaining >= 15 and time.time() < deadline:
            rng = np.random.default_rng(123)
            x_start3 = rng.uniform(-2, 2, dim)
            cd_calls3 = min(remaining, int(remaining * 0.5))
            run_coordinate_descent(x_start3, cd_calls3)

        # Phase 6: Final Nelder-Mead polish
        remaining = budget - calls_made
        if remaining >= 5 and x_best is not None and time.time() < deadline:
            nm_calls3 = min(remaining, 20)
            run_cmaes(x_best.copy(), nm_calls3)

        # Final safety: ensure we have a result
        if x_best is None:
            x_best = np.zeros(dim)
            f_best = evaluate(x_best)

        return x_best, f_best
# EVOLVE-BLOCK-END
