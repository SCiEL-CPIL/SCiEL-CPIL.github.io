"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead optimizer for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        
        # Strategy: Multi-start Nelder-Mead
        # We divide the budget into chunks for restarts.
        # Let's try 8 restarts with 50 calls each = 400 calls.
        # Remaining 100 calls for final polish on the best point.
        
        n_restarts = 8
        budget_per_restart = 50
        
        for i in range(n_restarts):
            # Random start point
            x0 = rng.uniform(-5, 5, dim)
            
            # Run Nelder-Mead
            x_opt, f_opt, calls_used = self._nelder_mead(func, x0, dim, budget_per_restart, rng)
            
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
                
        # Final polish on the best point found
        if best_x is not None:
            remaining_budget = budget - (n_restarts * budget_per_restart)
            # Ensure we don't exceed budget
            # _nelder_mead will stop if it hits the budget
            x_polished, f_polished, _ = self._nelder_mead(func, best_x, dim, remaining_budget, rng)
            if f_polished < best_f:
                best_f = f_polished
                best_x = x_polished.copy()
                
        # If no restarts were done or failed, return best found
        if best_x is None:
             # Fallback: just return a random point if something went wrong
             best_x = rng.uniform(-5, 5, dim)
             best_f = func(best_x)
             
        return best_x, best_f

    def _nelder_mead(self, func, x0, dim, budget, rng):
        x0 = np.clip(x0, -5, 5)
        simplex = [x0.copy()]
        simplex_f = [func(x0)]
        calls = 1
        
        step = 1.0
        for i in range(dim):
            if calls >= budget:
                break
            x_new = x0.copy()
            x_new[i] += step
            x_new = np.clip(x_new, -5, 5)
            simplex.append(x_new)
            f_val = func(x_new)
            simplex_f.append(f_val)
            calls += 1
            
        while calls < budget:
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            x_worst = simplex[-1]
            f_worst = simplex_f[-1]
            x_best = simplex[0]
            f_best = simplex_f[0]
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + (centroid - x_worst)
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            calls += 1
            
            if calls >= budget:
                if f_ref < f_worst:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                break
                
            if f_ref < simplex_f[0]:
                # Expansion
                x_exp = centroid + 2 * (centroid - x_worst)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                calls += 1
                
                if f_exp < f_ref:
                    x_new = x_exp
                    f_new = f_exp
                else:
                    x_new = x_ref
                    f_new = f_ref
            elif f_ref < simplex_f[-2]:
                x_new = x_ref
                f_new = f_ref
            else:
                # Contraction
                x_con = centroid + 0.5 * (x_worst - centroid)
                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                calls += 1
                
                if f_con < f_worst:
                    x_new = x_con
                    f_new = f_con
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        if calls >= budget:
                            break
                        simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = func(simplex[i])
                        calls += 1
                    if calls >= budget:
                        break
                    continue
            
            simplex[-1] = x_new
            simplex_f[-1] = f_new
            
        best_idx = np.argmin(simplex_f)
        return simplex[best_idx], simplex_f[best_idx], calls
# EVOLVE-BLOCK-END
