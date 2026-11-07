"""Optimizer for BBOB-style f17 (Schaffers F7) with multi-start and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-start optimizer combining global sampling with Nelder-Mead refinement.
    
    Strategy:
    1. Phase 1: Global exploration using stratified sampling to identify promising regions.
    2. Phase 2: Local refinement using Nelder-Mead from the best starting points.
    
    Includes time budget management to ensure completion within wall-clock limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 280.0  # Safety margin before 300s kill

        # Track best solution
        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_eval(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            try:
                fx = func(np.asarray(x, dtype=float))
                fx = float(fx)
                if np.isfinite(fx):
                    if fx < best_f:
                        best_f = fx
                        best_x = np.array(x, dtype=float)
                    return fx
                else:
                    return np.inf
            except Exception:
                call_count -= 1
                return np.inf

        # Phase 1: Global exploration with stratified sampling
        # Use a combination of random and grid-like points
        n_global = min(50, budget // 2)
        rng = np.random.default_rng(42)

        # Generate stratified points
        points = []
        
        # Add center and corners
        points.append(np.zeros(dim))
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 4.0
                points.append(p)
        
        # Add random points
        while len(points) < n_global:
            p = rng.uniform(-5, 5, dim)
            points.append(p)
        
        # Evaluate all global points
        for p in points:
            if call_count >= budget:
                break
            if time.time() - start_time > time_limit - 60:  # Reserve 60s for local
                break
            safe_eval(np.clip(p, -5, 5))

        if best_x is None:
            # Fallback: just return a random point
            x = rng.uniform(-5, 5, dim)
            safe_eval(x)
            return best_x, best_f

        # Phase 2: Local refinement using Nelder-Mead
        # Select top candidates for local search
        # Re-evaluate to get a sorted list (or track during Phase 1)
        # For simplicity, we'll use the best point and a few others
        
        # Collect evaluated points and their values
        evaluated = []
        for p in points:
            if call_count >= budget:
                break
            pc = np.clip(p, -5, 5)
            # Check if already evaluated (approximate)
            fx = safe_eval(pc)
            if np.isfinite(fx):
                evaluated.append((pc, fx))
        
        # Sort by fitness
        evaluated.sort(key=lambda x: x[1])
        
        # Try Nelder-Mead from top candidates
        n_local = min(5, len(evaluated))
        for i in range(n_local):
            if call_count >= budget:
                break
            if time.time() - start_time > time_limit - 10:
                break
            
            x0 = evaluated[i][0]
            f0 = evaluated[i][1]
            
            # Nelder-Mead options
            options = {
                'maxiter': max(10, (budget - call_count) // (n_local - i)),
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True
            }
            
            try:
                result = minimize(
                    lambda x: safe_eval(x),
                    x0,
                    method='Nelder-Mead',
                    options=options
                )
                if result.success or np.isfinite(result.fun):
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x
            except Exception:
                pass
        
        # Final check: ensure we have a valid result
        if best_x is None:
            x = rng.uniform(-5, 5, dim)
            safe_eval(x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
