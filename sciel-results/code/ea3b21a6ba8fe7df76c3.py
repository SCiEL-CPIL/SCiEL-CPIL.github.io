import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 270.0
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def run_nm(x0, max_iter, xatol=1e-8, fatol=1e-10):
            nonlocal calls
            if calls >= budget or time.time() >= deadline:
                return
            try:
                def objective(x):
                    return evaluate(x)
                
                minimize(
                    objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': xatol,
                        'fatol': fatol,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        # Phase 1: Diverse Exploration with L-BFGS-B for faster convergence
        num_exploration = 8
        calls_per_exploration = max(10, (budget - 100) // num_exploration)
        
        starts = [np.zeros(n)]
        for _ in range(num_exploration - 1):
            starts.append(rng.uniform(-4.5, 4.5, size=n))
            
        for x0 in starts:
            if calls >= budget or time.time() >= deadline:
                break
            
            # Try L-BFGS-B first (faster for smooth functions), then NM
            try:
                def lbfgsb_obj(x):
                    return evaluate(x)
                
                minimize(
                    lbfgsb_obj,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * n,
                    options={
                        'maxiter': max(10, calls_per_exploration // 2),
                        'maxls': 50
                    }
                )
            except Exception:
                pass
            
            # Follow up with NM for local polish
            if calls < budget and time.time() < deadline:
                run_nm(x0, calls_per_exploration // 2, xatol=1e-8, fatol=1e-10)

        # Phase 2: Polish with tighter tolerances
        if best_x is not None and calls < budget and time.time() < deadline:
            num_polish = 3
            calls_per_polish = max(10, (budget - calls) // num_polish)
            for _ in range(num_polish):
                if calls >= budget or time.time() >= deadline:
                    break
                x0_polish = best_x + rng.normal(0, 0.05, size=n)
                run_nm(x0_polish, calls_per_polish, xatol=1e-10, fatol=1e-12)

        return best_x, best_f
