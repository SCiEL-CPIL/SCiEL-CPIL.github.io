"""Rewritten program: Nelder-Mead optimizer."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead style optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initial simplex
        # Center at 0, step size 4.0 to cover [-5, 5]
        step = 4.0
        simplex = np.zeros((d + 1, d))
        for i in range(d):
            simplex[i + 1, i] = step
            
        # Evaluate initial simplex
        f_vals = np.array([func(x) for x in simplex])
        budget_used = d + 1
        
        best_x = simplex[np.argmin(f_vals)].copy()
        best_f = np.min(f_vals)
        
        # Main loop
        while budget_used < budget:
            # Sort simplex
            idx = np.argsort(f_vals)
            simplex = simplex[idx]
            f_vals = f_vals[idx]
            
            # Check convergence
            if np.max(f_vals) - np.min(f_vals) < 1e-10:
                break
                
            # Centroid (excluding worst)
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            budget_used += 1
            
            if f_r < f_vals[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                budget_used += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_vals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
            else:
                # Contraction
                if f_r < f_vals[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    budget_used += 1
                    
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_vals[i] = func(simplex[i])
                            budget_used += 1
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    budget_used += 1
                    
                    if f_c < f_vals[-1]:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_vals[i] = func(simplex[i])
                            budget_used += 1
                        
            # Update best
            current_best_idx = np.argmin(f_vals)
            if f_vals[current_best_idx] < best_f:
                best_f = f_vals[current_best_idx]
                best_x = simplex[current_best_idx].copy()
                
            if budget_used >= budget:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
