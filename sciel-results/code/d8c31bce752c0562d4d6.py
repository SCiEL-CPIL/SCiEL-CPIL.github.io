import numpy as np
import scipy.optimize as opt
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None
        self.deadline = 300.0 # Safety margin below 330s

    def __call__(self, func):
        self.start_time = time.time()
        rng = np.random.default_rng(42) # Fixed seed for reproducibility in local tests, though evaluator might vary
        
        # 1. Global Sampling Phase
        # Use Latin Hypercube Sampling for better coverage than pure random
        n_global = 40
        # LHS: divide each dimension into n_global intervals, pick one random point in each
        x_global = np.zeros((n_global, self.dim))
        for d in range(self.dim):
            # Create edges
            edges = np.linspace(-5, 5, n_global + 1)
            # Pick one random point in each interval
            points = rng.uniform(edges[:-1], edges[1:])
            # Shuffle to decorrelate dimensions
            points = points[rng.permutation(n_global)]
            x_global[:, d] = points
        
        f_vals = np.array([func(x) for x in x_global])
        
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            return x_global[np.argmin(f_vals)], f_vals.min()

        # Sort by function value
        sorted_idx = np.argsort(f_vals)
        
        # 2. Local Refinement Phase
        # Pick top K candidates
        k_candidates = 5
        top_idx = sorted_idx[:k_candidates]
        
        remaining_budget = self.budget - n_global
        if remaining_budget <= 0:
            return x_global[sorted_idx[0]], f_vals[sorted_idx[0]]
        
        budget_per_candidate = remaining_budget // k_candidates
        
        best_x = x_global[top_idx[0]]
        best_f = f_vals[top_idx[0]]
        
        for i in range(k_candidates):
            # Check deadline
            if time.time() - self.start_time > self.deadline:
                break
            
            x0 = x_global[top_idx[i]]
            f0 = f_vals[top_idx[i]]
            
            # Run Nelder-Mead
            # initial_simplex_size: default is 0.05 * |x_i| + 0.00025
            # In 5D, we want a reasonable start.
            # maxiter controls function evaluations roughly (not exactly 1:1, but close)
            # scipy's Nelder-Mead maxiter is the number of function evaluations allowed.
            
            # We need to be careful with maxiter. 
            # scipy.optimize.minimize(method='Nelder-Mead', options={'maxiter': budget_per_candidate})
            
            try:
                res = opt.minimize(func, x0, method='Nelder-Mead', 
                                   options={'maxiter': budget_per_candidate, 'xatol': 1e-4, 'fatol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
                
        return best_x, best_f
