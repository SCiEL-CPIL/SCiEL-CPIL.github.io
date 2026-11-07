"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead Simplex Optimizer for BBOB f5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        # Initialize simplex at the center of the box [-5, 5]^dim
        # Since the minimizer is in [-4, 4]^dim, center is a good start.
        x0 = np.zeros(dim)
        
        # Define initial simplex
        # v0 is x0
        # vi is x0 + delta * e_i
        delta = 1.0
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            xi[i] += delta
            simplex.append(xi)
        
        # Evaluate initial simplex
        # We must ensure we don't exceed budget. 
        # Initial evals: dim + 1
        if self.budget < dim + 1:
             # Fallback to simple random if budget is tiny (unlikely per contract)
             return self._fallback(func, start_time, deadline)

        fvals = np.array([func(x) for x in simplex])
        calls_made = dim + 1
        best_x = simplex[np.argmin(fvals)]
        best_f = fvals.min()
        
        # Nelder-Mead parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrink
        
        # We have budget - calls_made remaining
        remaining_budget = self.budget - calls_made
        
        # Nelder-Mead loop
        # Each iteration uses up to 2 evaluations (reflection + expansion/contraction)
        # or 1 (contraction/shrink). 
        # To be safe, we limit iterations such that worst-case evals <= remaining_budget.
        # Worst case per iteration is 2 evals.
        max_iterations = remaining_budget // 2
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Sort simplex by function value
            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = fvals[idx]
            
            x_best = simplex[0]
            f_best = fvals[0]
            x_worst = simplex[-1]
            f_worst = fvals[-1]
            
            # Update global best
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
                
            # Check for convergence
            if np.max(fvals) - np.min(fvals) < 1e-10:
                break
                
            # Compute centroid of all points except worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # 1. Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls_made += 1
            
            if f_r < f_best:
                # 2. Expansion
                x_e = centroid + gamma * (centroid - x_worst)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls_made += 1
                
                if f_e < f_r:
                    x_new, f_new = x_e, f_e
                else:
                    x_new, f_new = x_r, f_r
                    
                # Replace worst
                simplex[-1] = x_new
                fvals[-1] = f_new
                
            elif f_r < fvals[-2]: # f_r is better than 2nd worst
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
                
            else:
                # 3. Contraction
                # Outside contraction if f_r >= f_worst, else inside
                if f_r < f_worst:
                    # Inside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Outside contraction
                    x_c = centroid - rho * (centroid - x_worst)
                    
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls_made += 1
                
                if f_c < min(f_r, f_worst):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # 4. Shrink
                    # Shrink all points except best towards best
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls_made += 1
                        if calls_made > self.budget:
                             break
                    if calls_made > self.budget:
                         break
        
        return best_x, best_f

    def _fallback(self, func, start_time, deadline):
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        return x, fx
# EVOLVE-BLOCK-END
