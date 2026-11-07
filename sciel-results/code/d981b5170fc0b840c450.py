"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar


class Optimizer:
    """Brent-based coordinate descent with adaptive bracketing for step-ellipsoidal f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        calls = 0

        def eval_func(x):
            nonlocal calls
            calls += 1
            return float(func(np.asarray(x, dtype=float)))

        # Phase 1: Global coarse sampling to find a good starting region
        # Use stratified sampling across the box
        n_init = 15
        x_best = None
        fx_best = np.inf

        # Generate stratified points
        for i in range(n_init):
            # Mix of random and structured points
            if i < 5:
                x_cand = rng.uniform(-4.5, 4.5, d)
            elif i == 5:
                x_cand = np.zeros(d)
            elif i == 6:
                x_cand = rng.uniform(-2, 2, d)
            else:
                # Latin hypercube-style spread
                u = (i + rng.uniform(0, 1)) / n_init
                x_cand = -5 + 10 * u + rng.normal(0, 0.5, d)
                x_cand = np.clip(x_cand, -5, 5)

            fx = eval_func(x_cand)
            if fx < fx_best:
                fx_best = fx
                x_best = x_cand.copy()

        if calls >= budget:
            return x_best, fx_best

        # Phase 2: Brent-based coordinate descent with adaptive bounds
        # For each axis, we'll perform a 1D minimization using Brent's method
        # with a bounded search that adapts based on current best

        def minimize_axis_1d(x_base, axis, lo, hi, max_evals):
            """Minimize along one axis using bounded Brent method."""
            nonlocal calls

            if calls >= budget:
                return x_base, np.inf

            x_work = x_base.copy()
            current_val = x_work[axis]

            def g(t):
                x_work[axis] = t
                return eval_func(x_work)

            # Ensure we have enough budget
            if calls >= budget - 1:
                return x_base, np.inf

            try:
                res = minimize_scalar(
                    g,
                    bounds=(lo, hi),
                    method='bounded',
                    options={'maxiter': max_evals, 'xatol': 1e-6}
                )
                fx_new = float(res.fun)
                x_new = x_work.copy()
            except Exception:
                return x_base, np.inf

            if fx_new < fx_best - 1e-12:
                return x_new, fx_new
            else:
                return x_base, np.inf

        # Main coordinate descent loop
        # Start with wider max_iter, then reduce for refinement
        phases = [
            (12, 1.0),   # (max_iter_per_axis, bound_shrink_factor)
            (8, 0.5),
            (6, 0.25),
            (4, 0.1),
            (3, 0.05),
            (2, 0.02),
        ]

        for max_iter, shrink in phases:
            if calls >= budget - d * max_iter - 5:
                break

            # Determine bounds based on current best
            for axis in range(d):
                if calls >= budget - 3:
                    break

                # Adaptive bounds: start wide, then tighten around current best
                center = x_best[axis]
                if shrink >= 1.0:
                    lo = -5.0
                    hi = 5.0
                else:
                    # Tighten bounds around current best
                    half_width = 5.0 * shrink
                    lo = max(-5.0, center - half_width)
                    hi = min(5.0, center + half_width)

                    # Ensure minimum width for search
                    if hi - lo < 0.1:
                        lo = max(-5.0, center - 0.05)
                        hi = min(5.0, center + 0.05)

                x_new, fx_new = minimize_axis_1d(x_best, axis, lo, hi, max_iter)
                if fx_new < fx_best:
                    x_best = x_new
                    fx_best = fx_new

        # Phase 3: Fine-grained local search using small perturbations
        remaining = budget - calls
        if remaining > 10:
            # Try small random perturbations around best
            step_sizes = [0.1, 0.05, 0.02, 0.01]
            for step in step_sizes:
                if calls >= budget - 2:
                    break
                # Try a few random directions
                n_trials = min(5, (budget - calls) // 2)
                for _ in range(n_trials):
                    if calls >= budget - 1:
                        break
                    x_trial = x_best + rng.normal(0, step, d)
                    x_trial = np.clip(x_trial, -5, 5)
                    fx_trial = eval_func(x_trial)
                    if fx_trial < fx_best:
                        x_best = x_trial
                        fx_best = fx_trial

        return x_best, fx_best
# EVOLVE-BLOCK-END
