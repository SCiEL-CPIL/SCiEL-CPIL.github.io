"""Adaptive Coordinate Descent + Nelder-Mead optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Adaptive coordinate descent for rapid convergence + Nelder-Mead for refinement."""

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

        def run_adaptive_cd(x0, max_calls):
            """Adaptive coordinate descent: step size shrinks only on failure."""
            nonlocal calls_made, x_best, f_best
            if calls_made >= budget or time.time() > deadline:
                return

            x = np.array(x0, dtype=np.float64)
            f = evaluate(x)
            if f < f_best:
                f_best = f
                x_best = x.copy()

            step = 5.0  # Start large to cover the box
            min_step = 1e-6

            # We iterate until we run out of calls or time
            # Or until step becomes very small and no moves are made
            while calls_made < max_calls and time.time() < deadline:
                move_made = False
                for d in range(dim):
                    if calls_made >= max_calls or time.time() > deadline:
                        break
                    
                    # Try positive step
                    trial = x.copy()
                    trial[d] += step
                    trial = np.clip(trial, -5, 5)
                    f_trial = evaluate(trial)
                    if f_trial < f:
                        x = trial
                        f = f_trial
                        move_made = True
                    else:
                        # Try negative step if positive failed
                        trial = x.copy()
                        trial[d] -= step
                        trial = np.clip(trial, -5, 5)
                        f_trial = evaluate(trial)
                        if f_trial < f:
                            x = trial
                            f = f_trial
                            move_made = True

                if f < f_best:
                    f_best = f
                    x_best = x.copy()

                if not move_made:
                    step *= 0.5
                    if step < min_step:
                        break
                else:
                    # If a move was made, keep step size or slightly increase?
                    # Keeping it is safer. Shrink if we want to be aggressive?
                    # Standard adaptive: keep step if move made, shrink if not.
                    pass

        def run_nm(x0, max_calls):
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
                                        'xatol': 1e-8, 'fatol': 1e-10,
                                        'adaptive': True, 'initial_simplex': None})
                if res.fun < f_best:
                    f_best = float(res.fun)
                    x_best = np.array(res.x, dtype=np.float64)
            except Exception:
                pass

        # Phase 0: Minimal exploration to find a good basin
        # Büche-Rastrigin has minima at integers. 
        # We try origin and a few random points to ensure we don't start in a bad local min
        # that is far from the global one.
        rng = np.random.default_rng(42)
        candidates = [
            np.zeros(dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0
        ]
        
        # Evaluate candidates to pick the best starting point for CD
        for c in candidates:
            if calls_made >= budget or time.time() > deadline:
                break
            evaluate(c)

        # Phase 1: Adaptive Coordinate Descent
        # Use most of the budget here. 
        # CD is very efficient for smooth-ish functions and can navigate Rastrigin valleys
        # if the step size adapts well.
        cd_budget = int(budget * 0.8)
        if x_best is not None and time.time() < deadline:
            run_adaptive_cd(x_best.copy(), cd_budget)

        # Phase 2: Nelder-Mead Refinement
        # Use remaining budget for final polish
        remaining = budget - calls_made
        if remaining >= 10 and x_best is not None and time.time() < deadline:
            nm_calls = min(remaining, 50)
            run_nm(x_best.copy(), nm_calls)

        # Final safety
        if x_best is None:
            x_best = np.zeros(dim)
            f_best = evaluate(x_best)

        return x_best, f_best
# EVOLVE-BLOCK-END
