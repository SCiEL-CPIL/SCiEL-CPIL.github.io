# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _halton(index, base):
    f = 1.0
    r = 0.0
    while index > 0:
        f = f / base
        r = r + f * (index % base)
        index = index // base
    return r

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Halton sequence global search
        max_global_calls = int(self.budget * 0.8)
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        i = 1
        while calls_made < max_global_calls and time.time() < deadline:
            point = np.array([_halton(i, primes[d]) for d in range(self.dim)])
            x = point * 10.0 - 5.0
            safe_call(x)
            i += 1
            if i % 10 == 0 and time.time() > deadline:
                break

        # Phase 2: Nelder-Mead polish
        if calls_made < self.budget and time.time() < deadline:
            def nm_func(x):
                nonlocal best_x, best_f, calls_made
                if calls_made >= self.budget:
                    return best_f # Return current best to stop improvement? 
                    # Actually, returning a large value is better to signal stop?
                    # But if we return best_f, it might think it's stuck.
                    # Better to just let it run and rely on maxiter.
                x = np.clip(x, -5, 5)
                f = func(x)
                calls_made += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f

            remaining = self.budget - calls_made
            max_iter = max(10, remaining // (self.dim + 1))
            
            try:
                result = minimize(
                    nm_func,
                    x0=best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
