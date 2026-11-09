"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Derivative-free optimizer using Nelder-Mead (Simplex method)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Wrap func to count evaluations and handle clipping
        eval_count = [0]
        def wrapped_func(x):
            eval_count[0] += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return float(func(x))

        # Initial simplex
        # Start at a random point to avoid deterministic pitfalls
        x0 = rng.uniform(-4, 4, dim)
        
        # Initial step size for simplex construction
        step = 1.0 
        
        simplex = []
        fvals = []
        
        # First point
        simplex.append(x0.copy())
        fvals.append(wrapped_func(x0))
        
        # Other points
        for i in range(dim):
            xi = x0.copy()
            # Try a step in dimension i
            # Ensure it stays within bounds roughly
            xi[i] += step
            if xi[i] > 5.0:
                xi[i] = 5.0 - step
            simplex.append(xi.copy())
            fvals.append(wrapped_func(xi))
            
        best_x = simplex[0].copy()
        best_f = fvals[0]
        for i in range(len(simplex)):
            if fvals[i] < best_f:
                best_f = fvals[i]
                best_x = simplex[i].copy()

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        max_iter = budget # Upper bound on iterations, will break earlier if budget runs out
        
        for _ in range(max_iter):
            if eval_count[0] >= budget:
                break
                
            # Sort simplex by function value
            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]
            
            # Check convergence: if the spread of function values is very small
            if abs(fvals[0] - fvals[-1]) < 1e-10:
                break
                
            # Worst point
            x_worst = simplex[-1]
            f_worst = fvals[-1]
            
            # Centroid of all points except the worst
            x_centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_reflect = x_centroid + alpha * (x_centroid - x_worst)
            x_reflect = np.clip(x_reflect, -5.0, 5.0)
            f_reflect = wrapped_func(x_reflect)
            
            if eval_count[0] >= budget:
                break
                
            if f_reflect < fvals[0]:
                # Reflection is better than best, try expansion
                x_expand = x_centroid + gamma * (x_reflect - x_worst)
                x_expand = np.clip(x_expand, -5.0, 5.0)
                f_expand = wrapped_func(x_expand)
                
                if eval_count[0] >= budget:
                    break
                    
                if f_expand < f_reflect:
                    x_new = x_expand
                    f_new = f_expand
                else:
                    x_new = x_reflect
                    f_new = f_reflect
            elif f_reflect < fvals[-2]:
                # Reflection is between best and second worst
                x_new = x_reflect
                f_new = f_reflect
            else:
                # Reflection is worse than second worst, try contraction
                if f_reflect < f_worst:
                    # Outside contraction
                    x_contract = x_centroid + rho * (x_reflect - x_centroid)
                else:
                    # Inside contraction
                    x_contract = x_centroid - rho * (x_worst - x_centroid)
                    
                x_contract = np.clip(x_contract, -5.0, 5.0)
                f_contract = wrapped_func(x_contract)
                
                if eval_count[0] >= budget:
                    break
                    
                if f_contract < min(f_reflect, f_worst):
                    x_new = x_contract
                    f_new = f_contract
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fvals[i] = wrapped_func(simplex[i])
                        if eval_count[0] >= budget:
                            break
                    if eval_count[0] >= budget:
                        break
                    # Re-sort and continue
                    indices = np.argsort(fvals)
                    simplex = [simplex[i] for i in indices]
                    fvals = [fvals[i] for i in indices]
                    continue

            # Update simplex: replace worst point with new point
            simplex[-1] = x_new
            fvals[-1] = f_new
            
            # Update global best
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
            
            # Check for convergence in simplex size
            # If all points are very close, we might be stuck or converged
            # A simple check: if the range of coordinates is very small
            if np.all(np.abs(np.array(simplex) - simplex[0]) < 1e-6):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
