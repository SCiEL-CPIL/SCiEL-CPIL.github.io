"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead Simplex Optimizer with time-aware budget management."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin well before 330s limit
        
        n = self.dim
        budget = self.budget
        
        # Nelder-Mead parameters
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma = 0.5    # Shrink
        
        # Initialize simplex
        # Start at center, expand slightly to ensure non-degeneracy
        x0 = np.zeros(n)
        f0 = func(x0)
        
        # Initial step size
        # Heuristic: small step to avoid immediate clipping issues but large enough to explore
        step = 0.5 
        
        simplex = [x0.copy()]
        f_simplex = [f0]
        
        for i in range(n):
            xi = x0.copy()
            xi[i] += step
            # Clip to box [-5, 5] to respect constraints
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            simplex.append(xi)
            f_simplex.append(fi)
            
        calls_made = n + 1
        best_x = simplex[0]
        best_f = f0
        
        # Track best for scoring
        # We need to return the best x found. The problem statement says 
        # "return value of __call__ is ignored", but we should still maintain 
        # a valid state. Wait, the contract says:
        # "The return value of __call__ is ignored. More than 500 calls... makes the whole result invalid"
        # It also says: "return the best result found" in the directive for deadline.
        # However, the seed program returns (x, fx). I will stick to that interface.
        
        # Main loop
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(f_simplex)
            simplex = [simplex[i] for i in order]
            f_simplex = [f_simplex[i] for i in order]
            
            # Update global best
            if f_simplex[0] < best_f:
                best_f = f_simplex[0]
                best_x = simplex[0].copy()
                
            # Calculate centroid of all points except the worst
            x_worst = simplex[-1]
            f_worst = f_simplex[-1]
            
            # Centroid of the n best points
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls_made += 1
            
            if f_r < f_simplex[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - x_worst)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls_made += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_simplex[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
            elif f_r < f_simplex[-2]:
                # Simple reflection
                simplex[-1] = x_r
                f_simplex[-1] = f_r
            else:
                # Contraction
                # Check if it's an outside or inside contraction
                if f_r < f_worst:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - x_worst)
                
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls_made += 1
                
                if f_c < min(f_r, f_worst):
                    simplex[-1] = x_c
                    f_simplex[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_simplex[i] = func(simplex[i])
                        calls_made += 1
                        
                        if calls_made >= budget:
                            break
                    if calls_made >= budget:
                        break
            
            # Optional: Terminate if simplex is too small
            # Calculate spread
            spread = np.max(np.abs(np.array(simplex) - simplex[0]), axis=0)
            if np.max(spread) < 1e-8:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
