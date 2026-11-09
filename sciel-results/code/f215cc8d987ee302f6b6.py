"""
Optimizer for Bent Cigar (f12) combining Sobol global search with 
Adaptive Nelder-Mead local refinement.
"""
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State tracking
        calls_used = 0
        best_x = None
        best_fx = np.inf
        
        # Objective wrapper to count calls and track best
        def wrapped_obj(x):
            nonlocal calls_used, best_x, best_fx
            if time.time() >= deadline or calls_used >= budget:
                raise Exception("Limit exceeded")
            
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = x.reshape(-1)
            
            calls_used += 1
            
            # Evaluate
            fx = func(x)
            
            # Update best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            return fx

        # --- Phase 1: Global Search with Sobol Sequence ---
        # Use a Sobol sequence for low-discrepancy sampling
        # 32 points is sufficient to locate the basin in 5D
        n_global = 32
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        samples = sobol.random(n_global)
        
        # Map [0, 1]^d to [-5, 5]^d
        samples = samples * 10.0 - 5.0
        
        # Evaluate all global samples
        for x in samples:
            if time.time() >= deadline or calls_used >= budget:
                break
            try:
                wrapped_obj(x)
            except Exception:
                break
        
        # If we didn't find any point (shouldn't happen), return center
        if best_x is None:
            x0 = np.zeros(dim)
            wrapped_obj(x0)
        else:
            x0 = best_x.copy()
            
        # --- Phase 2: Local Refinement with Adaptive Nelder-Mead ---
        # Standard Nelder-Mead can be slow on anisotropic functions if the simplex
        # is not aligned. We modify it to track per-dimension step sizes.
        
        # Initialize Simplex
        # Start with a small perturbation around x0
        initial_step = 0.5
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        for i in range(dim):
            simplex[i+1] = x0.copy()
            simplex[i+1][i] += initial_step
            
        # Evaluate initial simplex
        fx_simplex = np.zeros(dim + 1)
        for i in range(dim + 1):
            try:
                fx_simplex[i] = wrapped_obj(simplex[i])
            except Exception:
                # If we ran out of budget/time during init, we stop here
                return best_x, best_fx

        # Per-dimension adaptive step sizes
        # sigma[i] represents the "width" of the search in dimension i
        sigma = np.full(dim, initial_step)
        
        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma_nm = 0.5 # Shrink (for simplex shrinkage, not to be confused with step size)
        
        # Main Loop
        max_iter = 1000
        for _ in range(max_iter):
            if time.time() >= deadline or calls_used >= budget:
                break
            
            # Sort simplex by function value
            order = np.argsort(fx_simplex)
            simplex = simplex[order]
            fx_simplex = fx_simplex[order]
            
            # Check convergence
            if np.max(fx_simplex) - np.min(fx_simplex) < 1e-12:
                break
            
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst_point = simplex[-1]
            
            # Reflection
            x_r = centroid + alpha * (centroid - worst_point)
            x_r = np.clip(x_r, -5, 5)
            try:
                fx_r = wrapped_obj(x_r)
            except Exception:
                break
            
            if fx_r < fx_simplex[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - worst_point)
                x_e = np.clip(x_e, -5, 5)
                try:
                    fx_e = wrapped_obj(x_e)
                except Exception:
                    break
                
                if fx_e < fx_r:
                    simplex[-1] = x_e
                    fx_simplex[-1] = fx_e
                    # Success: Increase step sizes in directions of the move
                    move_dir = x_e - worst_point
                    for i in range(dim):
                        if abs(move_dir[i]) > 1e-8:
                            sigma[i] *= 1.1
                else:
                    simplex[-1] = x_r
                    fx_simplex[-1] = fx_r
                    # Partial Success: Moderate increase
                    move_dir = x_r - worst_point
                    for i in range(dim):
                        if abs(move_dir[i]) > 1e-8:
                            sigma[i] *= 1.05
            elif fx_r < fx_simplex[-2]:
                # Accept Reflection
                simplex[-1] = x_r
                fx_simplex[-1] = fx_r
                # Small increase
                move_dir = x_r - worst_point
                for i in range(dim):
                    if abs(move_dir[i]) > 1e-8:
                        sigma[i] *= 1.02
            else:
                # Contraction
                # Outside contraction
                x_oc = centroid + 0.5 * (worst_point - centroid)
                x_oc = np.clip(x_oc, -5, 5)
                try:
                    fx_oc = wrapped_obj(x_oc)
                except Exception:
                    break
                
                if fx_oc < fx_simplex[-1]:
                    simplex[-1] = x_oc
                    fx_simplex[-1] = fx_oc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        try:
                            fx_simplex[i] = wrapped_obj(simplex[i])
                        except Exception:
                            break
                    # Failure: Decrease step sizes
                    sigma *= 0.9
                    if np.min(sigma) < 1e-8:
                        break
            
            # Clamp step sizes to reasonable bounds
            sigma = np.clip(sigma, 1e-8, 5.0)

        return best_x, best_fx
