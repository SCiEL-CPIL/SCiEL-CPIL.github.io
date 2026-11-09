import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Random Exploration
        # Use 10% of budget for exploration
        n_explore = max(10, budget // 10)
        
        best_x = None
        best_f = float('inf')
        
        for i in range(n_explore):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        # If no exploration done or failed, fallback to center
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        # Phase 2: Nelder-Mead Local Optimization
        # Initialize simplex around best_x
        x0 = best_x
        f0 = best_f
        
        # Simplex initialization
        # Size of the simplex
        simplex_size = 0.5
        simplex = [x0 + simplex_size * np.eye(dim)[i] for i in range(dim)]
        simplex.append(x0) # Center
        
        # Evaluate simplex
        f_vals = [func(x) for x in simplex]
        
        # Track best
        for i, f in enumerate(f_vals):
            if f < best_f:
                best_f = f
                best_x = simplex[i].copy()
                
        evals_used = n_explore + len(simplex)
        remaining_budget = budget - evals_used
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        while evals_used < budget and time.time() < deadline:
            # Sort simplex by function value
            order = np.argsort(f_vals)
            simplex = [simplex[i] for i in order]
            f_vals = [f_vals[i] for i in order]
            
            # Check for convergence or stagnation
            if len(set([round(f, 10) for f in f_vals])) == 1:
                break
                
            x_worst = simplex[-1]
            f_worst = f_vals[-1]
            x_best = simplex[0]
            f_best = f_vals[0]
            
            # Centroid of all but worst
            x_centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_reflected = x_centroid + alpha * (x_centroid - x_worst)
            x_reflected = np.clip(x_reflected, -5, 5)
            f_reflected = func(x_reflected)
            evals_used += 1
            
            if f_reflected < f_best:
                # Expansion
                x_expanded = x_centroid + gamma * (x_reflected - x_centroid)
                x_expanded = np.clip(x_expanded, -5, 5)
                f_expanded = func(x_expanded)
                evals_used += 1
                
                if f_expanded < f_reflected:
                    simplex[-1] = x_expanded
                    f_vals[-1] = f_expanded
                else:
                    simplex[-1] = x_reflected
                    f_vals[-1] = f_reflected
            elif f_reflected < f_vals[-2]:
                # Accept reflection
                simplex[-1] = x_reflected
                f_vals[-1] = f_reflected
            else:
                # Contraction
                x_contracted = x_centroid + rho * (x_worst - x_centroid)
                x_contracted = np.clip(x_contracted, -5, 5)
                f_contracted = func(x_contracted)
                evals_used += 1
                
                if f_contracted < f_worst:
                    simplex[-1] = x_contracted
                    f_vals[-1] = f_contracted
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_vals[i] = func(simplex[i])
                        evals_used += 1
                        
            # Update global best
            for i, f in enumerate(f_vals):
                if f < best_f:
                    best_f = f
                    best_x = simplex[i].copy()
                    
        return best_x, best_f
