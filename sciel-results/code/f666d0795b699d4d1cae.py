"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead optimizer with random restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # We will perform multiple restarts
        # Let's decide on number of restarts based on budget
        # For n=5, simplex has 6 points.
        # Let's aim for ~150 evaluations per restart, so 3 restarts.
        num_restarts = 3
        budget_per_restart = budget // num_restarts
        
        global_best_x = None
        global_best_f = np.inf
        
        rng = np.random.default_rng(42)
        
        for restart in range(num_restarts):
            # Initial center for this restart
            x0 = rng.uniform(-4, 4, n)
            
            # Initialize simplex
            # Use a small step size for initial simplex to avoid immediate boundary issues, 
            # but large enough to explore.
            # Standard Nelder-Mead initializes with step size 0.05 * |x_i| or 0.05 if x_i is 0.
            # Here we are in [-5, 5], so 0.05 is too small. Let's use 0.5 or 1.0.
            # Actually, for a box of size 10, a step of 1.0 is reasonable.
            step = 1.0
            
            simplex = np.zeros((n + 1, n))
            for i in range(n):
                simplex[i] = x0
                simplex[i][i] += step
            simplex[n] = x0 - step # Or just x0 + step in last dim? 
            # Standard: simplex[i] = x0 + step * e_i for i<n, simplex[n] = x0 - step * e_0? 
            # Let's use a standard initialization:
            # v_0 = x0
            # v_i = x0 + step * e_i for i=1..n
            # This creates a simplex of n+1 vertices.
            
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(1, n + 1):
                simplex[i] = x0
                simplex[i][i-1] += step
            
            # Evaluate initial simplex
            f_values = np.zeros(n + 1)
            calls_used = 0
            
            for i in range(n + 1):
                f_values[i] = func(simplex[i])
                calls_used += 1
                if f_values[i] < global_best_f:
                    global_best_f = f_values[i]
                    global_best_x = simplex[i].copy()
            
            remaining_budget = budget_per_restart - calls_used
            
            # Nelder-Mead parameters
            alpha = 1.0    # Reflection
            gamma = 2.0    # Expansion
            rho = 0.5      # Contraction
            sigma = 0.5    # Shrinkage
            
            max_iter = remaining_budget # Each iteration uses at least 1 call
            
            iter_count = 0
            while remaining_budget > 0 and iter_count < max_iter:
                # Sort vertices by function value
                order = np.argsort(f_values)
                simplex = simplex[order]
                f_values = f_values[order]
                
                # Check for convergence (optional, but budget is main constraint)
                # if np.std(f_values) < 1e-10: break
                
                # Calculate centroid of all points except the worst
                centroid = np.mean(simplex[:n], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[n])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_used += 1
                remaining_budget -= 1
                
                if f_r < global_best_f:
                    global_best_f = f_r
                    global_best_x = x_r.copy()
                
                if f_values[0] <= f_r < f_values[n-1]:
                    # Reflection is good
                    simplex[n] = x_r
                    f_values[n] = f_r
                elif f_r < f_values[0]:
                    # Reflection is better than best, try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    if remaining_budget <= 0:
                        break
                    f_e = func(x_e)
                    calls_used += 1
                    remaining_budget -= 1
                    
                    if f_e < global_best_f:
                        global_best_f = f_e
                        global_best_x = x_e.copy()
                    
                    if f_e < f_r:
                        simplex[n] = x_e
                        f_values[n] = f_e
                    else:
                        simplex[n] = x_r
                        f_values[n] = f_r
                else:
                    # Reflection is worse than worst, try contraction
                    if f_r < f_values[n]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[n])
                    
                    x_c = np.clip(x_c, -5, 5)
                    if remaining_budget <= 0:
                        break
                    f_c = func(x_c)
                    calls_used += 1
                    remaining_budget -= 1
                    
                    if f_c < global_best_f:
                        global_best_f = f_c
                        global_best_x = x_c.copy()
                    
                    if f_c < min(f_r, f_values[n]):
                        simplex[n] = x_c
                        f_values[n] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if remaining_budget <= 0:
                                break
                            f_values[i] = func(simplex[i])
                            calls_used += 1
                            remaining_budget -= 1
                            
                            if f_values[i] < global_best_f:
                                global_best_f = f_values[i]
                                global_best_x = simplex[i].copy()
                        
                        if remaining_budget <= 0:
                            break
                        
                        # Re-sort after shrink
                        order = np.argsort(f_values)
                        simplex = simplex[order]
                        f_values = f_values[order]
                
                iter_count += 1
        
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
