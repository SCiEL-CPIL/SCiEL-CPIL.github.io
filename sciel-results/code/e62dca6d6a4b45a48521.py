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
            if f < f_best:
                f_best = f
                x_best = x.copy()

            # Start with large step, shrink geometrically
            step_sizes = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125,
                          0.015625, 0.0078125, 0.00390625, 0.001953125,
                          0.0009765625, 0.00048828125]

            for step in step_sizes:
                if calls_made >= max_calls or time.time() > deadline:
                    break
                # Sweep each dimension in both directions
                for d in range(dim):
                    if calls_made >= max_calls or time.time() > deadline:
                        break
                    # Try positive direction
                    trial = x.copy()
                    trial[d] += step
                    trial = np.clip(trial, -5, 5)
                    f_trial = evaluate(trial)
                    if f_trial < f:
                        x = trial
                        f = f_trial
                    # Try negative direction
                    trial = x.copy()
                    trial[d] -= 2 * step
                    trial = np.clip(trial, -5, 5)
                    f_trial = evaluate(trial)
                    if f_trial < f:
                        x = trial
                        f = f_trial

            if f < f_best:
                f_best = f
                x_best = x.copy()

        def run_nelder_mead(x0, max_calls, xatol=1e-8, fatol=1e-10):
            """Run Nelder-Mead for local refinement."""
            nonlocal calls_made, x_best, f_best
            if x0 is None or max_calls < 10 or time.time() > deadline:
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
                    if fx < f_best:
                        f_best = fx
                        x_best = x.copy()
                return fx

            try:
                res = minimize(wrapped, local_x, method='Nelder-Mead',
                               options={'maxiter': max_calls, 'maxfev': max_calls,
                                        'xatol': xatol, 'fatol': fatol,
                                        'adaptive': True, 'initial_simplex': None})
                if res.fun < f_best:
                    f_best = float(res.fun)
                    x_best = np.array(res.x, dtype=np.float64)
            except Exception:
                pass

        def run_polish(x0, max_calls):
            """High-precision polish with very tight tolerances."""
            nonlocal calls_made, x_best, f_best
            if x0 is None or max_calls < 5 or time.time() > deadline:
                return
            run_nelder_mead(x0, max_calls, xatol=1e-12, fatol=1e-14)

        # Phase 1: Multi-scale coordinate descent from origin
        cd_calls = int(budget * 0.5)
        if time.time() < deadline:
            run_coordinate_descent(np.zeros(dim), cd_calls)

        # Phase 2: Nelder-Mead refinement on best point
        remaining = budget - calls_made
        if remaining >= 10 and x_best is not None and time.time() < deadline:
            nm_calls = min(remaining, 60)
            run_nelder_mead(x_best.copy(), nm_calls)

        # Phase 3: Second coordinate descent from a different starting point
        remaining = budget - calls_made
        if remaining >= 20 and time.time() < deadline:
            rng = np.random.default_rng(42)
            x_start2 = rng.uniform(-3, 3, dim)
            cd_calls2 = min(remaining, int(remaining * 0.6))
            run_coordinate_descent(x_start2, cd_calls2)

        # Phase 4: Small random restart to escape local minima, then refine
        remaining = budget - calls_made
        if remaining >= 10 and x_best is not None and time.time() < deadline:
            # Try a few small perturbations around best point
            rng = np.random.default_rng(777)
            for _ in range(3):
                if calls_made >= budget or time.time() > deadline:
                    break
                perturbation = rng.normal(0, 0.05, dim)
                x_trial = np.clip(x_best + perturbation, -5, 5)
                f_trial = evaluate(x_trial)
                if f_trial < f_best:
                    x_best = x_trial.copy()
            nm_calls2 = min(remaining - calls_made, 40)
            if nm_calls2 >= 10:
                run_nelder_mead(x_best.copy(), nm_calls2)

        # Phase 5: Third coordinate descent
        remaining = budget - calls_made
        if remaining >= 15 and time.time() < deadline:
            rng = np.random.default_rng(123)
            x_start3 = rng.uniform(-2, 2, dim)
            cd_calls3 = min(remaining, int(remaining * 0.7))
            run_coordinate_descent(x_start3, cd_calls3)

        # Phase 6: Final high-precision polish
        remaining = budget - calls_made
        if remaining >= 5 and x_best is not None and time.time() < deadline:
            polish_calls = min(remaining, 30)
            run_polish(x_best.copy(), polish_calls)

        # Final safety
        if x_best is None:
            x_best = np.zeros(dim)
            f_best = evaluate(x_best)

        return x_best, f_best
# EVOLVE-BLOCK-END
