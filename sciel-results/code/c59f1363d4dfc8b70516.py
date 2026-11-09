import numpy as np
import time
import sys
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._start_time = None
        self._deadline = None
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        self._start_time = time.time()
        self._deadline = self._start_time + 325  # 5s margin
        self._call_count = 0
        self._best_x = np.zeros(self.dim)
        self._best_f = np.inf

        def safe_call(x):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                return 1e10
            self._call_count += 1
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        # Phase 1: Differential Evolution
        # Budget 500. DE with popsize=15, maxiter=30 -> 15 + 30*15 = 465 calls.
        # Leaves 35 calls for polishing.
        try:
            bounds = [(-5.0, 5.0)] * self.dim
            res = differential_evolution(
                safe_call, 
                bounds, 
                maxiter=20, 
                popsize=20, 
                tol=1e-6, 
                mutation=(0.5, 1.0), 
                recombination=0.7,
                seed=42,
                polish=False  # Disable built-in polish to control it ourselves
            )
            # Update best from res (safe_call already did, but ensure consistency)
            if res.fun < self._best_f:
                self._best_f = res.fun
                self._best_x = res.x.copy()
        except Exception:
            pass

        # Phase 2: Nelder-Mead Polish
        # Use remaining budget (approx 35 calls)
        if self._call_count < self.budget - 5 and time.time() < self._deadline:
            try:
                res_nm = minimize(
                    safe_call,
                    self._best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                # safe_call updates self._best_f and self._best_x automatically
            except Exception:
                pass

        return self._best_x, self._best_f
