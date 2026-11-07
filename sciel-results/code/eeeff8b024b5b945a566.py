import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 270.0
        n = self.dim
        budget = self.budget

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

        def run_nm(x0, max_iter, xatol=1e-10, fatol=1e-12):
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

        # Phase 1: Dense Sobol quasi-random exploration
        # Use Sobol sequence to systematically cover [-5, 5]^n
        sobol_sampler = qmc.Sobol(d=n, scramble=True, seed=42)
        
        # Allocate ~40% of budget to Sobol exploration
        sobol_budget = max(100, int(budget * 0.4))
        
        # Generate Sobol points
        sobol_points = sobol_sampler.random(sobol_budget)
        # Map from [0, 1]^n to [-5, 5]^n
        sobol_points = sobol_points * 10.0 - 5.0
        
        for i, x in enumerate(sobol_points):
            if calls >= budget or time.time() >= deadline:
                break
            evaluate(x)
        
        # Phase 2: Local refinement around best point found
        if best_x is not None and calls < budget and time.time() < deadline:
            remaining = budget - calls
            
            # Run Nelder-Mead from the global best
            run_nm(best_x.copy(), max(50, remaining // 2), xatol=1e-12, fatol=1e-14)
            
            # Additional refinements from nearby points
            if best_x is not None and calls < budget and time.time() < deadline:
                remaining = budget - calls
                rng = np.random.default_rng(123)
                for _ in range(min(3, remaining // 30)):
                    if calls >= budget or time.time() >= deadline:
                        break
                    # Small perturbation around current best
                    x0 = best_x + rng.normal(0, 0.1, size=n)
                    run_nm(x0, max(20, (budget - calls) // 4), xatol=1e-12, fatol=1e-14)
        
        return best_x, best_f
