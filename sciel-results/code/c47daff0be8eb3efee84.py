"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid Random Search + Nelder-Mead Optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        rng = np.random.default_rng(0)
        
        # 1. Global Exploration: Random Search
        # Use ~10% of budget for random sampling to find a good basin
        n_random = max(10, self.budget // 10)
        
        best_x = None
        best_f = float('inf')
        
        for _ in range(n_random):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x
        
        if best_x is None:
            # Fallback if no calls were made (unlikely but safe)
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            
        calls_made = n_random
        remaining_budget = self.budget - calls_made
        
        if remaining_budget <= 0:
            return best_x, best_f

        # 2. Local Exploitation: Nelder-Mead
        # Setup initial simplex
        # Start from best_x, expand simplex with a small step
        step = 0.5 
        simplex = np.zeros((self.dim + 1, self.dim))
        simplex[0] = best_x
        
        for i in range(self.dim):
            simplex[i+1] = best_x.copy()
            simplex[i+1][i] += step
        
        # Evaluate initial simplex
        f_simplex = np.array([func(v) for v in simplex])
        calls_made += self.dim + 1
        
        if calls_made >= self.budget:
             # Find best in simplex
             idx = np.argmin(f_simplex)
             return simplex[idx], f_simplex[idx]

        # Nelder-Mead Parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        
        # We will implement a manual Nelder-Mead to control budget and time precisely
        # instead of relying on scipy's minimize which might not stop exactly on budget or time.
        
        x0 = simplex[0]
        fx0 = f_simplex[0]
        
        # Sort simplex by function value
        order = np.argsort(f_simplex)
        simplex = simplex[order]
        f_simplex = f_simplex[order]
        
        # Main Loop
        while calls_made < self.budget:
            if time.time() > deadline:
                break
                
            # 1. Reflection
            centroid = np.mean(simplex[:-1], axis=0)
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls_made += 1
            
            if f_r < f_simplex[0]:
                # 2. Expansion
                if f_r < f_simplex[-2]: # Check if it's better than second best? Standard NM checks if f_r < f_0
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_made += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                else:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
            elif f_r < f_simplex[-2]:
                # Accept reflection
                simplex[-1] = x_r
                f_simplex[-1] = f_r
            else:
                # 3. Contraction
                if f_r < f_simplex[-1]:
                    # Outside Contraction
                    x_c = centroid + rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_made += 1
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_simplex[i] = func(simplex[i])
                            calls_made += 1
                            if calls_made >= self.budget:
                                break
                else:
                    # Inside Contraction
                    x_c = centroid - rho * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_made += 1
                    if f_c < f_simplex[-1]:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_simplex[i] = func(simplex[i])
                            calls_made += 1
                            if calls_made >= self.budget:
                                break
                
                if calls_made >= self.budget:
                    break

            # Re-sort simplex
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]

            # Check for convergence (optional, but helps if we have budget left)
            # If the spread is very small, we might be stuck, but NM handles this via shrink.
            
        # Return the best point found
        best_idx = np.argmin(f_simplex)
        return simplex[best_idx], f_simplex[best_idx]

# EVOLVE-BLOCK-END
