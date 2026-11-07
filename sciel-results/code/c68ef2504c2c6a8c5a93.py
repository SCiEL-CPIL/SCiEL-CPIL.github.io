"""Combined optimizer: L-BFGS-B primary with multi-start and Nelder-Mead polish."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """L-BFGS-B based optimizer for Rosenbrock with multi-start and NM polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Phase 1: Initial Sampling
        # Use a mix of structured and random points to cover the space
        # The minimizer is in [-4, 4]^5
        n_structured = 10
        structured_points = []
        # Origin and small vectors
        structured_points.append(np.zeros(self.dim))
        for scale in [0.5, 1.0, 2.0, -0.5, -1.0, -2.0]:
            structured_points.append(np.ones(self.dim) * scale)
        # Axis-aligned points
        for i in range(self.dim):
            p = np.zeros(self.dim)
            p[i] = 1.0
            structured_points.append(p)
            
        # Trim if necessary, though 10 is small
        structured_points = structured_points[:n_structured]
        
        n_random = 40
        random_points = rng.uniform(-5, 5, (n_random, self.dim))
        
        all_points = np.vstack([structured_points, random_points])
        n_initial = len(all_points)
        
        # Evaluate all initial points
        fvals = np.array([func(x) for x in all_points])
        calls_used = n_initial
        
        # Sort points by function value
        sorted_indices = np.argsort(fvals)
        
        best_x = all_points[sorted_indices[0]].copy()
        best_f = fvals[sorted_indices[0]]
        
        if time.time() > deadline - 10:
            return best_x, best_f

        remaining = self.budget - calls_used
        
        # Phase 2: Multi-start L-BFGS-B
        # We will run L-BFGS-B from the top 2 best initial points
        # to avoid getting stuck in a bad local basin.
        
        n_restarts = 2
        # Allocate budget: 60% to first, 30% to second, 10% reserved for NM
        budget_per_restart = [int(remaining * 0.6), int(remaining * 0.3)]
        
        for i in range(n_restarts):
            if time.time() > deadline - 20:
                break
            if remaining <= 0:
                break
            
            start_idx = sorted_indices[i]
            start_x = all_points[start_idx].copy()
            
            # Cap the budget for this run to ensure we leave room for others/NM
            max_fun = min(budget_per_restart[i], remaining)
            
            call_count = [0]
            
            def lbfgs_wrapper(x, cc=call_count, mf=max_fun):
                if cc[0] >= mf:
                    raise ValueError("Budget exceeded")
                if time.time() > deadline - 5:
                    raise ValueError("Time exceeded")
                cc[0] += 1
                return func(x)
            
            try:
                res = minimize(
                    lbfgs_wrapper, 
                    start_x, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'maxfun': max_fun, 'maxls': 20, 'gtol': 1e-8}
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass
            
            remaining -= call_count[0]
            
            if time.time() > deadline - 15:
                break

        # Phase 3: Nelder-Mead Polish
        # NM is derivative-free and can navigate the narrow valley well
        # if started close to the minimum.
        if remaining > 20 and time.time() < deadline - 10:
            nm_budget = min(remaining, 50) # Cap NM at 50 calls to be safe
            nm_count = [0]
            
            def nm_wrapper(x):
                if nm_count[0] >= nm_budget:
                    raise ValueError("NM budget exceeded")
                if time.time() > deadline - 5:
                    raise ValueError("Time exceeded")
                nm_count[0] += 1
                return func(x)
            
            try:
                res_nm = minimize(
                    nm_wrapper, 
                    best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': nm_budget, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.copy()
                    
            except Exception:
                pass

        return best_x, best_f
