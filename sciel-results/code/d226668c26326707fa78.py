# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Portfolio optimizer for BBOB f7 (Step ellipsoidal) on [-5,5]^d."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Reset state
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

        def count_eval(x):
            self._call_count += 1
            val = func(x)
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        # The function is convex and smooth (C1).
        # L-BFGS-B is very effective.
        # Powell is robust to conditioning.
        # Nelder-Mead is robust and simple.
        
        # Strategy:
        # 1. Run L-BFGS-B from origin.
        # 2. Run Powell from origin.
        # 3. Run Nelder-Mead from origin.
        # 4. Polish the best result with L-BFGS-B.
        # 5. If budget remains, try random restarts.

        bounds = [(-5, 5)] * self.dim
        
        # 1. L-BFGS-B from origin
        if self._call_count < self.budget - 10 and time.time() < deadline:
            try:
                res = minimize(count_eval, np.zeros(self.dim), method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                pass

        # 2. Powell from origin
        if self._call_count < self.budget - 10 and time.time() < deadline:
            try:
                res = minimize(count_eval, np.zeros(self.dim), method='Powell',
                               bounds=bounds, options={'maxiter': 50, 'xtol': 1e-12, 'ftol': 1e-12})
            except Exception:
                pass

        # 3. Nelder-Mead from origin
        if self._call_count < self.budget - 10 and time.time() < deadline:
            try:
                res = minimize(count_eval, np.zeros(self.dim), method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-12})
            except Exception:
                pass

        # 4. Polish the best found so far with L-BFGS-B
        if self._best_x is not None and self._call_count < self.budget - 10 and time.time() < deadline:
            try:
                res = minimize(count_eval, self._best_x, method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-10})
            except Exception:
                pass

        # 5. Random restarts if budget remains
        # The function is convex, so if we haven't converged, we might be in a bad spot or just need more iterations.
        # But since it's convex, we should have converged. 
        # If not, maybe the initial methods failed or were too slow.
        # Try a few random points.
        
        n_restarts = 5
        for _ in range(n_restarts):
            if self._call_count >= self.budget - 20:
                break
            if time.time() > deadline:
                break
            
            x0 = rng.uniform(-4, 4, self.dim)
            try:
                res = minimize(count_eval, x0, method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': 20, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                continue

        # Final polish if we have a bit more budget
        if self._best_x is not None and self._call_count < self.budget - 5 and time.time() < deadline:
            try:
                res = minimize(count_eval, self._best_x, method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': 10, 'ftol': 1e-15, 'gtol': 1e-12})
            except Exception:
                pass

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
