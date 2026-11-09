"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES exploration + Nelder-Mead / coordinate-descent polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        start = time.time()
        deadline = start + 290.0  # safety margin before 330s kill

        calls = 0

        def f(x):
            nonlocal calls
            calls += 1
            return float(func(x))

        # ---------- Phase A: adaptive (1+1)-ES for fast basin finding ----------
        x = rng.uniform(-5, 5, dim)
        fx = f(x)
        best_x, best_fx = x.copy(), fx
        sigma = 2.0

        n_explore = max(20, int(0.55 * budget))  # use most of budget for search
        for i in range(n_explore):
            if calls >= budget - 5 or time.time() > deadline - 20:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = f(y)
            if fy < fx:
                x, fx = y, fy
                if fx < best_fx:
                    best_fx, best_x = fx, x.copy()
                sigma = min(sigma * 1.25, 8.0)
            else:
                sigma *= 0.88
                sigma = max(sigma, 1e-4)

        # ---------- Phase B: Nelder-Mead polish (scipy) ----------
        try:
            from scipy.optimize import minimize
            if calls < budget - 3 and time.time() < deadline - 5:
                res = minimize(
                    lambda v: f(np.clip(v, -5, 5)),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11,
                             'initial_simplex': best_x + 0.2 * np.ones(dim)},
                )
                if res.fun < best_fx:
                    best_fx, best_x = res.fun, np.clip(res.x, -5, 5).copy()
        except Exception:
            pass

        # ---------- Phase C: coordinate-descent polish for final precision ----------
        if calls < budget - 2 and time.time() < deadline - 3:
            step = 0.1
            for _ in range(40):
                if calls >= budget or time.time() > deadline - 1:
                    break
                improved = False
                for d in range(dim):
                    for sgn in (1.0, -1.0):
                        trial = best_x.copy()
                        trial[d] = np.clip(best_x[d] + sgn * step, -5, 5)
                        ft = f(trial)
                        if ft < best_fx:
                            best_x, best_fx = trial.copy(), ft
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-7:
                        break

        return best_x, best_fx
# EVOLVE-BLOCK-END
