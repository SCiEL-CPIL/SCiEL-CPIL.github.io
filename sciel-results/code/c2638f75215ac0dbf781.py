"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer for Katsuura (f23) on [-5,5]^5.
    
    Strategy:
    1. Quick random sampling to find a good initial point.
    2. Coordinate-wise refinement exploiting Katsuura's separable structure.
    3. Fine-tuning with small random perturbations.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf
        
        def evaluate(x):
            nonlocal calls, best_x, best_f
            calls += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Phase 1: Random sampling to find a good starting point
        # Use a portion of budget for initial exploration
        n_initial = min(50, budget // 5)
        
        # Start with center and some random points
        x0 = np.zeros(dim)
        f0 = evaluate(x0)
        
        # Try some structured points
        for i in range(1, n_initial):
            if time.time() > deadline:
                break
            # Mix of random and structured points
            if i % 2 == 0:
                # Random uniform
                x = np.random.uniform(-5, 5, dim)
            else:
                # Random but biased toward center
                x = np.random.uniform(-3, 3, dim)
            evaluate(x)
        
        # Phase 2: Coordinate-wise refinement (exploits separability)
        # Katsuura is separable: f(x) = sum_i f_i(x_i)
        # So we can optimize each coordinate independently given others
        
        x = best_x.copy()
        fx = best_f
        
        # Number of refinement iterations
        n_refine = 5
        for refine_iter in range(n_refine):
            if time.time() > deadline or calls >= budget:
                break
            
            # For each coordinate, do a fine search
            for d in range(dim):
                if time.time() > deadline or calls >= budget:
                    break
                
                # Current value
                x_cur = x.copy()
                f_cur = fx
                
                # Search range for this coordinate
                # Start with a reasonable range and shrink
                if refine_iter == 0:
                    half_range = 5.0
                else:
                    half_range = 5.0 / (2 ** refine_iter)
                
                # Try points in the neighborhood
                # Use a small grid or random sampling in the neighborhood
                n_local = min(10, (budget - calls) // (dim * (n_refine - refine_iter)))
                if n_local < 2:
                    n_local = 2
                
                for _ in range(n_local):
                    if time.time() > deadline or calls >= budget:
                        break
                    
                    # Sample in the neighborhood of current best for this coordinate
                    x_new = x_cur.copy()
                    x_new[d] = x_cur[d] + np.random.uniform(-half_range, half_range)
                    x_new = np.clip(x_new, -5, 5)
                    
                    f_new = evaluate(x_new)
                    if f_new < f_cur:
                        x_cur = x_new
                        f_cur = f_new
                
                # Update x and fx if improved
                if f_cur < fx:
                    x = x_cur
                    fx = f_cur
        
        # Phase 3: Fine-tuning with small random perturbations
        if calls < budget:
            sigma = 0.1
            n_fine = min(budget - calls, 100)
            for _ in range(n_fine):
                if time.time() > deadline or calls >= budget:
                    break
                
                x_new = x + sigma * np.random.standard_normal(dim)
                x_new = np.clip(x_new, -5, 5)
                f_new = evaluate(x_new)
                
                if f_new < fx:
                    x = x_new
                    fx = f_new
                    sigma *= 1.2
                else:
                    sigma *= 0.8
        
        return best_x, best_f
# EVOLVE-BLOCK-END
