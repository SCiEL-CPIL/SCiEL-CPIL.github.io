"""Portfolio-based search for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Deterministic Multi-Resolution Local Search (Grid Descent) for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit. 280s is a safe margin to ensure we return before kill.
        deadline = start_time + 280.0
        
        d = self.dim
        budget = self.budget
        
        # Global best tracking
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        # Random number generator
        rng = np.random.default_rng(42)
        
        # Phase 1: Coarse Global Search
        # We perform a structured grid search over a few dimensions at a time,
        # or a random grid to cover the space.
        # For f7 (step ellipsoidal), the landscape has large flat regions.
        # A random grid is more efficient than a full grid in 5D.
        # Let's use a set of random points to find a good basin.
        
        n_init = 20
        for i in range(n_init):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break
                
            x = rng.uniform(-4.5, 4.5, d)
            fx = func(x)
            calls_used += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if best_x is None:
            # Fallback if no calls were made (should not happen)
            x = np.zeros(d)
            best_f = func(x)
            calls_used += 1
            best_x = x.copy()
            
        # Phase 2: Deterministic Multi-Resolution Local Search
        # This is the core structural change.
        # Instead of stochastic (1+1)-ES, we use a deterministic coordinate-wise
        # search with adaptive step sizes.
        # We try to refine the current best point by moving along each coordinate
        # axis with a step size that starts large and shrinks.
        
        # We will iterate through a sequence of step sizes.
        # For each step size, we try to move in +/- direction along each coordinate.
        # If we find an improvement, we accept it and continue.
        # If we don't find an improvement in a full pass, we reduce the step size.
        
        # Initial step size
        step = 1.0
        min_step = 1e-4
        
        # We will loop until budget or time is up, or until step < min_step
        while calls_used < budget and step > min_step:
            if time.time() > deadline:
                break
                
            improved = False
            
            # Try each coordinate
            for i in range(d):
                if time.time() > deadline:
                    break
                if calls_used >= budget:
                    break
                    
                # Try positive step
                x_plus = best_x.copy()
                x_plus[i] += step
                x_plus = np.clip(x_plus, -5.0, 5.0)
                if not np.allclose(x_plus, best_x): # Avoid redundant call if clipped to same point
                    f_plus = func(x_plus)
                    calls_used += 1
                    if f_plus < best_f:
                        best_f = f_plus
                        best_x = x_plus.copy()
                        improved = True
                        # If improved, restart the coordinate loop? Or continue?
                        # Restarting ensures we don't miss a better direction in the same pass.
                        # But for efficiency, let's just continue.
                        # Actually, for coordinate descent, restarting is often better.
                        # Let's break out of the coordinate loop and restart the step size loop?
                        # No, just restart the coordinate loop for the current step size.
                        break
                
                # Try negative step
                x_minus = best_x.copy()
                x_minus[i] -= step
                x_minus = np.clip(x_minus, -5.0, 5.0)
                if not np.allclose(x_minus, best_x):
                    f_minus = func(x_minus)
                    calls_used += 1
                    if f_minus < best_f:
                        best_f = f_minus
                        best_x = x_minus.copy()
                        improved = True
                        break
                        
            if improved:
                # If we improved, we might want to keep the same step size for a bit
                # or reduce it slightly to refine.
                # Let's reduce it slightly to allow for finer tuning.
                step *= 0.9
            else:
                # No improvement in this pass, reduce step size significantly
                step *= 0.5
                
        # Phase 3: Final Stochastic Refinement
        # If we still have budget, use a small (1+1)-ES to polish.
        if calls_used < budget:
            sigma = step * 0.5
            if sigma < 1e-4:
                sigma = 1e-4
            current_x = best_x.copy()
            current_f = best_f
            stagnation = 0
            stagnation_threshold = 10
            
            while calls_used < budget:
                if time.time() > deadline:
                    break
                    
                z = rng.standard_normal(d)
                y = current_x + sigma * z
                y = np.clip(y, -5.0, 5.0)
                
                fy = func(y)
                calls_used += 1
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    current_x = y.copy()
                    current_f = fy
                    stagnation = 0
                    sigma *= 1.1
                else:
                    stagnation += 1
                    if fy == current_f:
                        sigma *= 1.2
                    else:
                        sigma *= 0.9
                        
                    if stagnation > stagnation_threshold:
                        # Restart local search
                        new_x = best_x + rng.standard_normal(d) * sigma * 2
                        new_x = np.clip(new_x, -5.0, 5.0)
                        new_f = func(new_x)
                        calls_used += 1
                        if new_f < best_f:
                            best_f = new_f
                            best_x = new_x.copy()
                        current_x = new_x.copy()
                        current_f = new_f
                        sigma = max(sigma, 1e-3)
                        stagnation = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
