"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Portfolio optimizer: L-BFGS-B local optimization for BBOB f20 (Schwefel)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        def count_wrapper(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)
        
        # Generate diverse starting points for multi-start
        # Mix of random points and points biased toward center
        n_starts = 8
        starts = []
        
        # 4 uniform random
        for i in range(4):
            x = rng.uniform(-4, 4, dim)
            starts.append(x)
        
        # 2 biased toward center
        for i in range(2):
            x = rng.uniform(-2, 2, dim)
            starts.append(x)
        
        # 2 more random with different range
        for i in range(2):
            x = rng.uniform(-4.5, 4.5, dim)
            starts.append(x)
        
        # Evaluate all starts to find best initial point
        start_fvals = []
        for x in starts:
            fval = count_wrapper(x)
            start_fvals.append(fval)
        
        # Sort starts by function value
        order = np.argsort(start_fvals)
        
        # Allocate budget: use ~300 calls for L-BFGS-B on best start,
        # remaining for additional local optimizations
        main_budget = 250
        remaining_budget = budget - len(starts) - main_budget
        
        # Run L-BFGS-B on the best start
        x0 = starts[order[0]]
        f0 = start_fvals[order[0]]
        
        if f0 < best_f:
            best_f = f0
            best_x = x0.copy()
        
        # L-BFGS-B with bounds
        bounds = [(-5, 5)] * dim
        
        try:
            result = minimize(
                count_wrapper,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'maxfun': main_budget}
            )
            
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
        
        # Use remaining budget for additional local optimizations on other starts
        calls_used_so_far = calls_made
        remaining_calls = budget - calls_used_so_far
        
        for idx in order[1:]:
            if remaining_calls <= 20:
                break
            if time.time() > deadline:
                break
            
            x0 = starts[idx]
            f0 = start_fvals[idx]
            
            if f0 >= best_f:
                continue
            
            # Allocate smaller budget for secondary optimizations
            alloc_budget = min(remaining_calls - 5, 80)
            if alloc_budget < 20:
                break
            
            try:
                result = minimize(
                    count_wrapper,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'maxfun': alloc_budget}
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
            
            remaining_calls = budget - calls_made
        
        # Final refinement: run L-BFGS-B from best found point with small budget
        if best_x is not None and budget - calls_made > 10 and time.time() < deadline:
            try:
                result = minimize(
                    count_wrapper,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 30, 'maxfun': min(50, budget - calls_made)}
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
