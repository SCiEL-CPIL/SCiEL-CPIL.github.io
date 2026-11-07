"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer for Büche-Rastrigin f4 on [-5,5]^5.
    
    Combines:
    1. Structured exploration (integer lattice + random) to find good basins
    2. Multi-start (1+1)-ES with adaptive step size for robust local refinement
    3. Final high-precision polish pass
    
    The Büche-Rastrigin function has periodic local minima. After hidden 
    rotation/shift, the structure is preserved but transformed. We exploit 
    this by first sampling integer lattice points (likely near local minima 
    in the original frame), then using adaptive local search to refine.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline

        # Phase 1: Structured Exploration
        # Sample integer lattice points and random continuous points
        # to identify good basins for local refinement
        
        n_explore = min(60, budget // 3)
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        budget_used = 0
        
        # Evaluate integer lattice points (likely near local minima)
        n_lattice = n_explore // 2
        for _ in range(n_lattice):
            if budget_used >= budget or not time_ok():
                break
            pt = rng.integers(-4, 5, size=dim).astype(float)
            pt = np.clip(pt, -5, 5)
            f_val = func(pt)
            budget_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()
        
        # Evaluate random continuous points
        n_random = n_explore - n_lattice
        for _ in range(n_random):
            if budget_used >= budget or not time_ok():
                break
            pt = np.clip(rng.uniform(-5, 5, dim), -5, 5)
            f_val = func(pt)
            budget_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()
        
        # Phase 2: Multi-Start (1+1)-ES with Adaptive Step Size
        # Use multiple starting points from the exploration phase
        # and run adaptive local search from each
        
        # Collect candidate starting points
        candidates = [best_x.copy()]
        
        # Add a few more random points as additional starting candidates
        n_extra_starts = min(5, max(1, (budget - budget_used) // 10))
        for _ in range(n_extra_starts):
            if budget_used >= budget or not time_ok():
                break
            pt = np.clip(rng.uniform(-5, 5, dim), -5, 5)
            candidates.append(pt)
        
        # Run (1+1)-ES from each candidate
        sigma_init = 1.5
        min_sigma = 0.001
        max_sigma = 10.0
        window_size = 10
        
        for start_pt in candidates:
            if budget_used >= budget or not time_ok():
                break
            
            x = start_pt.copy()
            fx = best_f if np.array_equal(x, best_x) else func(x)
            if not np.array_equal(x, best_x):
                budget_used += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            sigma = sigma_init
            success_window = []
            
            # Run (1+1)-ES from this starting point
            while budget_used < budget and time_ok():
                # Generate trial point
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                budget_used += 1
                
                # Update best and track success
                if fy < fx:
                    x, fx = y, fy
                    success_window.append(1)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    success_window.append(0)
                
                # Adapt sigma based on 1/5th success rule
                if len(success_window) >= window_size:
                    success_rate = np.mean(success_window[-window_size:])
                    if success_rate > 0.2:
                        sigma *= 1.05
                    elif success_rate < 0.1:
                        sigma *= 0.95
                    if len(success_window) > window_size:
                        success_window.pop(0)
                    sigma = max(min_sigma, min(sigma, max_sigma))
                
                # If we've converged (very small sigma and no recent improvement),
                # break to try next starting point
                if sigma < 0.01 and len(success_window) >= window_size:
                    if np.mean(success_window[-window_size:]) < 0.05:
                        break
        
        # Phase 3: Final High-Precision Polish
        # If budget remains, do a fine-grained local search around the best point
        
        remaining = budget - budget_used
        if remaining > 10 and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.1  # Start slightly larger to allow escape, then shrink
            min_sigma = 1e-6
            max_sigma = 1.0
            window_size = 5
            success_window = []
            
            for _ in range(remaining):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                budget_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    success_window.append(1)
                else:
                    success_window.append(0)
                
                # Adapt sigma
                if len(success_window) >= window_size:
                    success_rate = np.mean(success_window[-window_size:])
                    if success_rate > 0.3:
                        sigma *= 1.2
                    elif success_rate < 0.1:
                        sigma *= 0.8
                    if len(success_window) > window_size:
                        success_window.pop(0)
                    sigma = max(min_sigma, min(sigma, max_sigma))
        
        return best_x, best_f
# EVOLVE-BLOCK-END
