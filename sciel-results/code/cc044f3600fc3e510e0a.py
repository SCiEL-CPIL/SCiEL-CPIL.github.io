"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with adaptive sigma + Nelder-Mead polish, time-guarded."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return 1e18
            x_clip = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            calls_made += 1
            return float(func(x_clip))

        def time_left():
            return (deadline - time.time()) > 5.0

        # Phase 1: (1+1)-ES for fast early progress
        rng = np.random.default_rng(42)
        x = rng.uniform(-4.0, 4.0, self.dim)
        fx = safe_func(x)
        best_x, best_f = x.copy(), fx
        sigma = 1.5
        es_budget = int(self.budget * 0.7)  # use 70% for ES

        for i in range(es_budget):
            if not time_left() or calls_made >= self.budget:
                break
            step = sigma * rng.standard_normal(self.dim)
            y = np.clip(x + step, -5.0, 5.0)
            fy = safe_func(y)
            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_x, best_f = x.copy(), fx
                sigma *= 1.25
            else:
                sigma *= 0.85
            # Clamp sigma
            sigma = np.clip(sigma, 1e-8, 10.0)

        # Phase 2: Nelder-Mead polish from best point
        if time_left() and calls_made < self.budget and best_f < np.inf:
            try:
                nm_budget = self.budget - calls_made
                # Limit NM function evaluations
                maxfev = min(nm_budget, 200)
                res = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxfev': maxfev,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    },
                )
                if res.fun < best_f:
                    best_x = np.clip(res.x, -5.0, 5.0)
                    best_f = float(res.fun)
            except Exception:
                pass

        # Phase 3: If time and budget remain, try a second NM from a perturbed best
        if time_left() and calls_made < self.budget and best_f < np.inf:
            try:
                remaining = self.budget - calls_made
                if remaining > 30:
                    # Perturb best point slightly
                    rng2 = np.random.default_rng(123)
                    x2 = np.clip(best_x + 0.1 * rng2.standard_normal(self.dim), -5.0, 5.0)
                    f2 = safe_func(x2)
                    if f2 < best_f:
                        best_x, best_f = x2.copy(), f2
                    res2 = minimize(
                        safe_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxfev': min(remaining, 150),
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True,
                        },
                    )
                    if res2.fun < best_f:
                        best_x = np.clip(res2.x, -5.0, 5.0)
                        best_f = float(res2.fun)
            except Exception:
                pass

        # Phase 4: Final tight polish if budget remains
        if time_left() and calls_made < self.budget and best_f < np.inf:
            try:
                remaining = self.budget - calls_made
                if remaining > 10:
                    res3 = minimize(
                        safe_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxfev': min(remaining, 100),
                            'xatol': 1e-12,
                            'fatol': 1e-14,
                            'adaptive': True,
                        },
                    )
                    if res3.fun < best_f:
                        best_x = np.clip(res3.x, -5.0, 5.0)
                        best_f = float(res3.fun)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
