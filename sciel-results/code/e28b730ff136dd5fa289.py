"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer using local search (Powell) on a hypervolume-contribution surrogate.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.ref_point = np.array([3000.0, 0.06])

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Helper to evaluate and track budget
        eval_count = 0
        max_budget = self.budget
        
        def eval_point(x):
            nonlocal eval_count
            if eval_count >= max_budget:
                return None
            eval_count += 1
            return problem(x)

        # 1. Initial Random Sampling
        # Use a small fraction of budget for initialization
        n_init = 10
        if n_init > max_budget:
            n_init = max_budget
        
        X_init = []
        F_init = []
        
        for _ in range(n_init):
            x = lo + (hi - lo) * np.random.random(self.dim)
            f = eval_point(x)
            if f is not None:
                X_init.append(x)
                F_init.append(f)
        
        if not X_init:
            return np.array([])

        X_init = np.array(X_init)
        F_init = np.array(F_init)

        # 2. Identify Non-Dominated Points
        def is_non_dominated(F):
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return ~dominated

        mask_nd = is_non_dominated(F_init)
        X_nd = X_init[mask_nd]
        F_nd = F_init[mask_nd]

        # 3. Local Search using Powell Method
        # Objective: Maximize Hypervolume Contribution
        # HV Contribution = (r1 - f1) * (r2 - f2) if f1 < r1 and f2 < r2, else 0 (penalized)
        
        def hv_surrogate(x):
            """
            Returns negative hypervolume contribution to be minimized by scipy.
            Penalizes points outside the reference box.
            """
            f = eval_point(x)
            if f is None:
                return 1e10 # Budget exhausted, return large penalty
            
            # Check if better than reference point
            if f[0] >= self.ref_point[0] or f[1] >= self.ref_point[1]:
                return 1e10 # Large penalty
            
            # Calculate contribution
            d1 = self.ref_point[0] - f[0]
            d2 = self.ref_point[1] - f[1]
            hv = d1 * d2
            
            return -hv # Minimize negative HV

        # Determine how many local searches to run
        # We have max_budget - eval_count evaluations left
        remaining_budget = max_budget - eval_count
        
        # Limit number of local searches to avoid running out of budget
        # Powell in 4D might take ~10-20 evals to converge reasonably
        # Let's allow up to 5 local searches if we have enough budget
        n_local = min(len(X_nd), 5)
        
        # If budget is tight, reduce n_local
        if n_local > 0:
            evals_per_search = max(1, remaining_budget // n_local)
        else:
            evals_per_search = 0

        if evals_per_search < 5:
            # If budget is too tight for meaningful local search, just return initial ND set
            # or do a few simple perturbations
            # For now, return what we have
            return X_nd

        X_all = list(X_nd)
        F_all = list(F_nd)

        # Run local search from each non-dominated starting point
        for i in range(n_local):
            x0 = X_nd[i]
            
            # Define bounds for the optimizer
            bounds_opt = list(zip(lo, hi))
            
            try:
                res = minimize(
                    hv_surrogate, 
                    x0, 
                    method='Powell',
                    bounds=bounds_opt,
                    options={'maxfev': evals_per_search, 'maxiter': 20, 'xatol': 1e-4, 'fatol': 1e-6}
                )
                
                if res.success or res.fun < 1e9: # If it didn't fail catastrophically
                    x_new = res.x
                    f_new = eval_point(x_new) # Ensure we have the exact evaluation for the final point
                    if f_new is not None:
                        X_all.append(x_new)
                        F_all.append(f_new)
            except Exception:
                continue

        if not X_all:
            return X_nd

        X_all = np.array(X_all)
        F_all = np.array(F_all)

        # 4. Final Non-Dominated Selection
        mask_final = is_non_dominated(F_all)
        X_result = X_all[mask_final]
        
        return X_result

# EVOLVE-BLOCK-END
