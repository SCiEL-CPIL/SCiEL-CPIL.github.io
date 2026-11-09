"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Sharp-ridge-aware optimizer: rapid axis detection + focused 1D search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None
        self.deadline = 285.0

    def __call__(self, func):
        self.start_time = time.time()
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Use a dict to hold mutable state for closures
        state = {
            'best_x': np.zeros(n),
            'best_f': np.inf,
            'calls': 0,
        }

        def _check_time():
            if self.start_time is None:
                return False
            return (time.time() - self.start_time) > self.deadline

        def call_func(x):
            """Call func, track calls and best. Returns fx or None if budget exceeded."""
            if state['calls'] >= budget or _check_time():
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = func(x)
            state['calls'] += 1
            if fx < state['best_f']:
                state['best_f'] = fx
                state['best_x'] = x.copy()
            return fx

        # ===== Phase 1: Rapid structured sampling =====
        center = np.zeros(n)
        fc = call_func(center)

        # Points along each axis at several radii
        axis_points = []
        for d in range(n):
            for r in [1.0, 2.0, 3.0, 4.0]:
                for s in [1.0, -1.0]:
                    x = np.zeros(n)
                    x[d] = s * r
                    axis_points.append(x)

        # Diagonal points
        diag_points = []
        for r in [1.5, 2.5, 3.5]:
            diag_points.append(np.ones(n) * r)
            diag_points.append(np.ones(n) * (-r))

        all_points = axis_points + diag_points
        rng.shuffle(all_points)

        for x in all_points:
            if state['calls'] >= budget or _check_time():
                break
            fx = call_func(x)
            if fx is None:
                break

        # ===== Phase 2: Estimate gradient direction from best point =====
        ref = state['best_x'].copy()
        step = 0.5
        dir_derivs = np.zeros(n)
        for d in range(n):
            if state['calls'] >= budget or _check_time():
                break
            xp = ref.copy()
            xp[d] = np.clip(xp[d] + step, -5.0, 5.0)
            xm = ref.copy()
            xm[d] = np.clip(xm[d] - step, -5.0, 5.0)
            f_p = call_func(xp)
            if f_p is None:
                break
            f_m = call_func(xm)
            if f_m is None:
                break
            denom = xp[d] - xm[d]
            if abs(denom) > 1e-12:
                dir_derivs[d] = (f_p - f_m) / denom

        # Find the sharpest direction
        abs_derivs = np.abs(dir_derivs)
        if n > 0 and np.max(abs_derivs) > 1e-10:
            sharp_dim = int(np.argmax(abs_derivs))
        else:
            sharp_dim = 0

        # ===== Phase 3a: 1D golden-section search along sharp_dim =====
        if state['calls'] < budget and not _check_time():
            gr = (np.sqrt(5) - 1) / 2
            lo = -5.0
            hi = 5.0
            x1 = lo + (1 - gr) * (hi - lo)
            x2 = lo + gr * (hi - lo)

            def f_1d(t):
                x = state['best_x'].copy()
                x[sharp_dim] = np.clip(t, -5.0, 5.0)
                return call_func(x)

            f1 = f_1d(x1)
            f2 = f_1d(x2)
            if f1 is not None and f2 is not None:
                for _ in range(40):
                    if state['calls'] >= budget or _check_time():
                        break
                    if f1 < f2:
                        hi = x2
                        x2 = x1
                        f2 = f1
                        x1 = lo + (1 - gr) * (hi - lo)
                        f1 = f_1d(x1)
                        if f1 is None:
                            break
                    else:
                        lo = x1
                        x1 = x2
                        f1 = f2
                        x2 = lo + gr * (hi - lo)
                        f2 = f_1d(x2)
                        if f2 is None:
                            break

        # ===== Phase 3b: Local refinement with L-BFGS-B =====
        if state['calls'] < budget and not _check_time():
            def wrapped_func(x):
                if state['calls'] >= budget or _check_time():
                    return state['best_f']
                x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                fx = func(x)
                state['calls'] += 1
                if fx < state['best_f']:
                    state['best_f'] = fx
                    state['best_x'] = x.copy()
                return fx

            bounds = [(-5.0, 5.0)] * n
            try:
                minimize(
                    wrapped_func,
                    state['best_x'].copy(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
                )
            except Exception:
                pass

        # ===== Phase 3c: Additional restarts if budget remains =====
        if state['calls'] < budget - 10 and not _check_time():
            bounds = [(-5.0, 5.0)] * n
            for _ in range(5):
                if state['calls'] >= budget - 10 or _check_time():
                    break
                x0 = state['best_x'] + rng.normal(0, 0.3, n)
                x0 = np.clip(x0, -5.0, 5.0)
                fx0 = call_func(x0)
                if fx0 is None:
                    break

                def wrapped_func2(x):
                    if state['calls'] >= budget or _check_time():
                        return state['best_f']
                    x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                    fx = func(x)
                    state['calls'] += 1
                    if fx < state['best_f']:
                        state['best_f'] = fx
                        state['best_x'] = x.copy()
                    return fx

                try:
                    minimize(
                        wrapped_func2,
                        x0.copy(),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12}
                    )
                except Exception:
                    pass

        return state['best_x'], state['best_f']
# EVOLVE-BLOCK-END
