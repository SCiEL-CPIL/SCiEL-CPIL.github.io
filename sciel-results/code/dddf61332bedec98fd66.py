import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        bounds = [(-5.0, 5.0)] * dim
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        # Helper to update best
        def update_best(x, f_val):
            nonlocal best_x, best_f
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x)
                
        # Phase 1: Global Search with Differential Evolution
        # DE is robust to rotation and shift.
        # We allocate ~300 evaluations to this phase.
        # popsize=10, maxiter=30 -> 300 evals
        try:
            res_de = differential_evolution(
                func, 
                bounds=bounds, 
                maxiter=30, 
                popsize=10,
                tol=1e-12,
                seed=42,
                polish=False,
                init='latinhypercube' # Better initial coverage
            )
            
            if hasattr(res_de, 'fun') and res_de.fun < best_f:
                update_best(res_de.x, res_de.fun)
                
        except Exception:
            pass # If DE fails, we fall back to best_x (which is 0,0... initially)
            # We should probably do a few random evals if DE fails immediately
            if best_f == float('inf'):
                for _ in range(10):
                    if time.time() > deadline: break
                    x = np.random.uniform(-5, 5, dim)
                    f = func(x)
                    update_best(x, f)

        # Check time
        if time.time() > deadline - 10:
            return best_x, best_f

        # Phase 2: Local Refinement with Nelder-Mead
        # Nelder-Mead is derivative-free and robust for final polish.
        # We allocate ~200 evaluations to this phase.
        try:
            res_nm = minimize(
                func, 
                best_x, 
                method='Nelder-Mead',
                options={
                    'maxiter': 100, 
                    'maxfev': 200, 
                    'xatol': 1e-9, 
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            
            if hasattr(res_nm, 'fun') and res_nm.fun < best_f:
                update_best(res_nm.x, res_nm.fun)
                
        except Exception:
            pass

        return best_x, best_f
