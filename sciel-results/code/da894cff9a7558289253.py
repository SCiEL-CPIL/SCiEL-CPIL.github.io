"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Adaptive Nelder-Mead with multi-start exploration."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin: 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Track global best
        global_best_x = None
        global_best_f = float('inf')
        
        # Track calls
        calls_made = 0
        
        # Helper to check time and budget
        def can_continue():
            nonlocal calls_made
            return calls_made < budget and (time.time() < deadline)
        
        # Nelder-Mead implementation
        def nelder_mead(func, x0, max_iter=100):
            nonlocal calls_made, global_best_x, global_best_f
            
            # Initial simplex
            simplex = [x0.copy()]
            for i in range(dim):
                new_point = x0.copy()
                if x0[i] != 0:
                    new_point[i] = 1.05 * x0[i]
                else:
                    new_point[i] = 0.00025
                simplex.append(new_point)
            
            # Evaluate initial simplex
            fvals = []
            for i in range(dim + 1):
                if not can_continue():
                    return x0, float('inf')
                f = func(simplex[i])
                calls_made += 1
                fvals.append(f)
                if f < global_best_f:
                    global_best_f = f
                    global_best_x = simplex[i].copy()
            
            fvals = np.array(fvals)
            
            # Nelder-Mead parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            for _ in range(max_iter):
                if not can_continue():
                    break
                
                # Sort simplex by function value
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = fvals[idx]
                
                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_made += 1
                if fr < global_best_f:
                    global_best_f = fr
                    global_best_x = xr.copy()
                
                if fr < fvals[0]:
                    # Try expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_made += 1
                    if fe < global_best_f:
                        global_best_f = fe
                        global_best_x = xe.copy()
                    
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Try contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_made += 1
                    if fc < global_best_f:
                        global_best_f = fc
                        global_best_x = xc.copy()
                    
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if not can_continue():
                                return simplex[0], fvals[0]
                            fvals[i] = func(simplex[i])
                            calls_made += 1
                            if fvals[i] < global_best_f:
                                global_best_f = fvals[i]
                                global_best_x = simplex[i].copy()
            
            # Return best point in simplex
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]
        
        # Multi-start strategy
        # Divide budget into exploration and exploitation phases
        # Phase 1: Global exploration with several random starts
        # Phase 2: Refinement of the best start
        
        # Number of random starts based on budget
        num_starts = min(10, max(3, budget // 50))
        
        # Initial random points
        rng = np.random.default_rng(42)
        
        # First, do a quick grid-like sampling to find a good region
        # For 5D, we can't do full grid, so use random sampling
        # Sample 20 points initially
        initial_samples = min(20, budget // 5)
        for i in range(initial_samples):
            if not can_continue():
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls_made += 1
            if f < global_best_f:
                global_best_f = f
                global_best_x = x.copy()
        
        # Now do Nelder-Mead from the best point found
        if global_best_x is not None and can_continue():
            remaining_budget = budget - calls_made
            # Allocate most of remaining budget to Nelder-Mead
            max_iter = min(200, remaining_budget // (dim + 1))
            nelder_mead(func, global_best_x, max_iter=max_iter)
        
        # If we still have budget, do a few more local refinements from other good points
        if can_continue():
            # Try a few more random starts with limited iterations
            additional_starts = min(5, (budget - calls_made) // 10)
            for i in range(additional_starts):
                if not can_continue():
                    break
                x = rng.uniform(-5, 5, dim)
                # Perturb around current best slightly
                x = 0.5 * global_best_x + 0.5 * x
                x = np.clip(x, -5, 5)
                
                f = func(x)
                calls_made += 1
                if f < global_best_f:
                    global_best_f = f
                    global_best_x = x.copy()
                
                # Quick local refinement
                if can_continue():
                    max_iter = min(50, (budget - calls_made) // (dim + 1))
                    if max_iter > 5:
                        nelder_mead(func, x, max_iter=max_iter)
        
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
