"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Black-box continuous optimizer for BBOB f22 (Gallagher 21 peaks).
    Uses a Multi-start (1+1)-Evolution Strategy.
    
    This approach divides the budget among several independent search processes.
    Each process uses a simple (1+1)-ES with adaptive step size.
    This is robust for multimodal landscapes and computationally efficient in low dimensions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Number of independent search threads
        n_starts = 10
        evals_per_start = budget // n_starts
        
        best_x = None
        best_f = np.inf
        n_evals = 0

        # Initialize starting points using a stratified random approach (Halton-like or just random)
        # Random is fine for 5D, but we can try to spread them out a bit
        starts = []
        for i in range(n_starts):
            # Random points in [-5, 5]^dim
            x0 = rng.uniform(-5, 5, size=dim)
            starts.append(x0)
        
        # Run each start
        for i in range(n_starts):
            if n_evals >= budget:
                break
                
            x = starts[i].copy()
            fx = func(x)
            n_evals += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Initial step size
            sigma = 1.0
            
            # (1+1)-ES loop for this start
            remaining_for_this = evals_per_start - 1
            # If this is the last start, use all remaining budget
            if i == n_starts - 1:
                remaining_for_this = budget - n_evals
            
            success_count = 0
            window_size = 10
            success_window = []
            
            for step in range(remaining_for_this):
                if n_evals >= budget:
                    break
                
                # Generate candidate
                z = rng.standard_normal(dim)
                x_new = x + sigma * z
                x_new = np.clip(x_new, -5, 5)
                
                fx_new = func(x_new)
                n_evals += 1
                
                # Update global best
                if fx_new < best_f:
                    best_f = fx_new
                    best_x = x_new.copy()
                
                # Accept if better
                if fx_new < fx:
                    x = x_new
                    fx = fx_new
                    success_window.append(1)
                else:
                    success_window.append(0)
                
                # Adapt sigma
                if len(success_window) >= window_size:
                    success_rate = np.mean(success_window)
                    # Simple adaptation rule
                    if success_rate > 0.5:
                        sigma *= 1.5
                    elif success_rate < 0.2:
                        sigma /= 1.5
                    # Keep sigma within reasonable bounds
                    sigma = np.clip(sigma, 1e-4, 10.0)
                    
                    # Reset window to avoid long-term bias
                    success_window = success_window[-window_size//2:] # Keep some history

        return best_x, best_f
# EVOLVE-BLOCK-END
