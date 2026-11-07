"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # Strategy: Multi-start Nelder-Mead
        # 4 restarts to balance global exploration and local exploitation
        num_restarts = 4
        
        for _ in range(num_restarts):
            if calls_made >= total_budget:
                break
                
            # Random start point, slightly inside the box
            start = rng.uniform(-4.5, 4.5, size=n)
            
            # Initialize simplex
            # Base point + n vertices along axes
            delta = 0.5
            simplex = np.zeros((n+1, n))
            simplex[0] = start
            for i in range(n):
                simplex[i+1] = start.copy()
                simplex[i+1][i] += delta
                
            # Ensure within bounds
            simplex = np.clip(simplex, -5, 5)
            
            # Evaluate initial simplex
            f_values = np.zeros(n+1)
            for i in range(n+1):
                if calls_made >= total_budget:
                    break
                f_values[i] = func(simplex[i])
                calls_made += 1
            
            if calls_made >= total_budget:
                break
                
            # Update best
            min_val = np.min(f_values)
            min_idx = np.argmin(f_values)
            if min_val < best_f:
                best_f = min_val
                best_x = simplex[min_idx].copy()
                
            # Nelder-Mead Loop
            while calls_made < total_budget:
                # Sort simplex by fitness
                order = np.argsort(f_values)
                simplex = simplex[order]
                f_values = f_values[order]
                
                # Convergence check
                if f_values[-1] - f_values[0] < 1e-10:
                    break
                    
                # Centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                
                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5, 5)
                
                if calls_made >= total_budget:
                    break
                f_r = func(x_r)
                calls_made += 1
                
                if f_values[0] <= f_r < f_values[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                elif f_r < f_values[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - worst)
                    x_e = np.clip(x_e, -5, 5)
                    
                    if calls_made >= total_budget:
                        break
                    f_e = func(x_e)
                    calls_made += 1
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                else:
                    # Contraction
                    beta = 0.5
                    if f_r < f_values[-1]:
                        # Outside contraction
                        x_c = centroid + beta * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - beta * (centroid - worst)
                    x_c = np.clip(x_c, -5, 5)
                    
                    if calls_made >= total_budget:
                        break
                    f_c = func(x_c)
                    calls_made += 1
                    
                    if f_c < min(f_r, f_values[-1]):
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n+1):
                            if calls_made >= total_budget:
                                break
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_values[i] = func(simplex[i])
                            calls_made += 1
                        
                        if calls_made >= total_budget:
                            break
                
                # Update global best
                min_val = np.min(f_values)
                min_idx = np.argmin(f_values)
                if min_val < best_f:
                    best_f = min_val
                    best_x = simplex[min_idx].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
