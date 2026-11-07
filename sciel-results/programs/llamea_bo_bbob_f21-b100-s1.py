"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Multi-start Local Optimization for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        self.calls = 0
        budget = self.budget
        dim = self.dim
        
        class StopSearch(Exception):
            pass

        def wrapped_func(x):
            self.calls += 1
            if self.calls > budget:
                raise StopSearch()
            return float(func(x))

        # Global Phase: Sobol sequence for initial points
        n_global = 40
        sampler = qmc.Sobol(d=dim, scramble=True)
        points = sampler.random(n_global)
        points = points * 10 - 5  # Map [0,1] to [-5,5]
        
        # Evaluate global points
        global_results = []
        for x in points:
            try:
                f_val = wrapped_func(x)
                global_results.append((f_val, x.copy()))
            except StopSearch:
                break
        
        if not global_results:
            return None, np.inf

        # Sort by fitness
        global_results.sort(key=lambda item: item[0])
        
        # Local Phase: Refine the best few points
        n_local_starts = 6
        start_points = [x for _, x in global_results[:n_local_starts]]
        
        best_x = global_results[0][1]
        best_f = global_results[0][0]
        
        bounds = [(-5, 5)] * dim
        
        for x0 in start_points:
            if self.calls >= budget - 10:
                break
            try:
                # Use L-BFGS-B for bounded local optimization
                res = minimize(wrapped_func, x0, method='L-BFGS-B', bounds=bounds, 
                              options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except StopSearch:
                break
            except Exception:
                continue
        
        # Second pass: Nelder-Mead on best few points for further refinement
        if self.calls < budget - 20:
            nm_starts = [x for _, x in global_results[:3]]
            # Also include any improved points from L-BFGS-B
            if best_x is not None:
                nm_starts.append(best_x)
            
            for x0 in nm_starts:
                if self.calls >= budget - 5:
                    break
                try:
                    res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                                  options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except StopSearch:
                    break
                except Exception:
                    continue
        
        return best_x, best_f
# EVOLVE-BLOCK-END
