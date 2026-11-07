import numpy as np
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    """Hybrid Global-Local Optimizer for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(-5, 5)] * self.dim
        
        # Allocate budget: 80% to Global Search (DE), 20% to Local Refinement (NM)
        # DE calls approx: popsize * (1 + maxiter)
        # We want DE calls <= 0.8 * budget
        de_target_calls = int(0.8 * self.budget)
        popsize = 10
        # maxiter such that 10 * (1 + maxiter) <= de_target_calls
        maxiter = max(1, int(de_target_calls / popsize) - 1)
        
        best_x = None
        best_f = np.inf
        
        # 1. Global Search with Differential Evolution
        try:
            res_de = differential_evolution(
                func,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-10,
                seed=0,
                init='latinhypercube',
                polish=False # We will polish manually
            )
            best_x = res_de.x
            best_f = res_de.fun
        except Exception:
            # Fallback: Random search
            best_x = np.random.uniform(-5, 5, self.dim)
            best_f = func(best_x)
            # If DE failed completely, we might have used few calls, but let's assume worst case
            # and just proceed to local search with remaining time/budget?
            # Actually, if exception happens early, we have lots of budget.
            # But handling partial failures is complex. 
            # Let's assume DE succeeds or fails fast.
            
        # 2. Local Refinement with Nelder-Mead
        # Estimate calls used by DE. 
        # If res_de exists, we can check nfev if available, else estimate.
        calls_used = 0
        if 'res_de' in locals() and hasattr(res_de, 'nfev'):
            calls_used = res_de.nfev
        else:
            calls_used = min(self.budget, popsize * (1 + maxiter))
            
        remaining_budget = max(0, self.budget - calls_used)
        
        if remaining_budget > 0 and best_x is not None:
            try:
                # Nelder-Mead is fast. Limit iterations to remaining budget.
                # Each iteration is 1 function call.
                res_nm = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining_budget,
                        'xatol': 1e-8,
                        'fatol': 1e-10
                    }
                )
                if res_nm.fun < best_f:
                    best_x = res_nm.x
                    best_f = res_nm.fun
            except Exception:
                pass

        return best_x, best_f
