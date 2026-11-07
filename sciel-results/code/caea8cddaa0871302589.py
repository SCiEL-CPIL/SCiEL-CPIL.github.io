"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead optimizer for BBOB f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s margin
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        # Track evaluations
        def eval_func(x):
            nonlocal calls_made
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            return f

        # Nelder-Mead implementation
        def nelder_mead(func_eval, x0, max_iter, step_size):
            n = self.dim
            # Initialize simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += step_size
            
            # Evaluate initial simplex
            fvals = np.array([func_eval(simplex[i]) for i in range(n + 1)])
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                    
                # Sort by function value
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + (centroid - simplex[-1])
                fr = func_eval(xr)
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[-1])
                    fe = func_eval(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    fc = func_eval(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = func_eval(simplex[i])
                            
                # Update global best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()
            
            return best_x, best_f

        # Multi-start strategy
        rng = np.random.default_rng(42)
        
        # Allocate budget among starts
        # Schwefel is rugged, so multiple starts are important
        # Let's try 10 starts with 50 evals each initially
        num_starts = 10
        evals_per_start = max(10, (self.budget - 1) // num_starts)
        
        for i in range(num_starts):
            if time.time() > deadline or calls_made >= self.budget:
                break
                
            # Random start point
            x0 = rng.uniform(-5, 5, self.dim)
            f0 = eval_func(x0)
            
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            # Run Nelder-Mead from this start
            # Limit iterations to avoid excessive time
            max_iter = min(evals_per_start, self.budget - calls_made)
            if max_iter <= 0:
                break
                
            # Adaptive step size based on remaining budget
            step_size = 1.0
            
            local_best_x, local_best_f = nelder_mead(
                eval_func, x0, max_iter, step_size
            )
            
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x
        
        # If we have budget left, do a final polish on the best point
        if calls_made < self.budget and best_x is not None:
            remaining = self.budget - calls_made
            if remaining > 5:
                # Small step refinement
                step_size = 0.5
                max_iter = min(remaining, 50)
                local_best_x, local_best_f = nelder_mead(
                    eval_func, best_x, max_iter, step_size
                )
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x
        
        return best_x, best_f

# EVOLVE-BLOCK-END
