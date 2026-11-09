import numpy as np
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Phase 1: Global search with Sobol
        n_global = 100
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # Sobol requires power of 2 for optimal properties, 128 is close to 100
        # We can just take the first 100
        points = sampler.random(128)[:n_global]
        
        F_global = []
        X_global = []
        
        for p in points:
            x = lo + (hi - lo) * p
            f = problem(x)
            F_global.append(f)
            X_global.append(x)
            
        F_global = np.array(F_global)
        X_global = np.array(X_global)
        
        # Find non-dominated points
        def is_dominated(f, F):
            # f is dominated if there exists g in F such that g <= f and g != f
            for g in F:
                if np.all(g <= f) and np.any(g < f):
                    return True
            return False
            
        nd_indices = []
        for i in range(len(F_global)):
            if not is_dominated(F_global[i], F_global):
                nd_indices.append(i)
                
        # Phase 2: Local search around best non-dominated points
        # Pick top 10 non-dominated points (by sum of objectives or just random sample of ND)
        if len(nd_indices) > 0:
            # Sort by hypervolume contribution or just sum
            # Simple: sort by sum of objectives
            sums = np.sum(F_global[nd_indices], axis=1)
            sorted_nd = np.argsort(sums)
            top_nd = [nd_indices[i] for i in sorted_nd[:10]]
        else:
            top_nd = [0] # Fallback
            
        budget_used = n_global
        remaining = self.budget - budget_used
        steps_per_point = remaining // max(1, len(top_nd))
        
        for idx in top_nd:
            x = X_global[idx].copy()
            f = F_global[idx].copy()
            
            for _ in range(steps_per_point):
                if budget_used >= self.budget:
                    break
                # Perturb
                # Use a small step size
                step = 0.1 * (hi - lo)
                direction = np.random.normal(0, 1, self.dim)
                x_new = x + direction * step
                x_new = np.clip(x_new, lo, hi)
                
                f_new = problem(x_new)
                budget_used += 1
                
                # If new point is better (non-dominated by old and improves some objective), accept
                # Simple acceptance: if sum is better
                if np.sum(f_new) < np.sum(f):
                    x = x_new
                    f = f_new
        
        return X_global
