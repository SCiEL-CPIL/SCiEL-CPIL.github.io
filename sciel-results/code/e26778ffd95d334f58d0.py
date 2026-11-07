"""Direct search pattern search with adaptive step size and random restarts for BBOB f18."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Pattern search with adaptive step size and sparse random restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        calls = [0]

        def call_func(x):
            calls[0] += 1
            return func(x)

        # Phase 1: Exploration
        # Sample random points to find a good starting region
        n_explore = min(30, budget // 10)
        
        best_x = np.zeros(dim)
        best_fx = float('inf')
        
        # Evaluate center first
        x0 = np.zeros(dim)
        fx0 = call_func(x0)
        best_x = x0.copy()
        best_fx = fx0

        for _ in range(n_explore - 1):
            if calls[0] >= budget:
                break
            x_try = rng.uniform(-5, 5, dim)
            fx_try = call_func(x_try)
            if fx_try < best_fx:
                best_fx = fx_try
                best_x = x_try.copy()

        # Phase 2: Pattern Search with binary search on step size
        x = best_x.copy()
        fx = best_fx
        
        # Directions: Orthogonal + Diagonal + Random
        directions = [np.eye(dim)[i] for i in range(dim)]
        # Add diagonal directions
        for i in range(dim - 1):
            d = np.zeros(dim)
            d[i] = 1.0
            d[i + 1] = 1.0
            directions.append(d)
        # Add some random directions
        for _ in range(dim):
            d = rng.normal(size=dim)
            norm = np.linalg.norm(d)
            if norm > 0:
                d /= norm
            directions.append(d)
            
        directions = np.array(directions)
        n_dirs = len(directions)
        
        # Step size management
        sigma = 1.0
        sigma_min = 1e-6
        sigma_max = 10.0
        
        # In each iteration, for each direction, use binary search to find best step
        while calls[0] < budget:
            improved_in_cycle = False
            
            for d in directions:
                if calls[0] >= budget:
                    break
                
                # Binary search for optimal step in direction d
                # Try both positive and negative directions
                best_x_dir = x
                best_fx_dir = fx
                best_step = 0.0
                
                # First, try a few coarse steps to find the sign and rough magnitude
                coarse_steps = [sigma, -sigma, 2*sigma, -2*sigma, 0.5*sigma, -0.5*sigma]
                for step in coarse_steps:
                    if calls[0] >= budget:
                        break
                    y = np.clip(x + step * d, -5, 5)
                    fy = call_func(y)
                    if fy < best_fx_dir:
                        best_fx_dir = fy
                        best_x_dir = y
                        best_step = step
                
                if best_step != 0.0:
                    # We found an improvement, now refine with binary search
                    # Determine the sign of the best step
                    sign = 1.0 if best_step > 0 else -1.0
                    # Binary search between 0 and the best step found
                    lo = 0.0
                    hi = abs(best_step)
                    best_hi_step = best_step
                    best_hi_fx = best_fx_dir
                    
                    # Do a few binary search iterations
                    for _ in range(3):
                        if calls[0] >= budget:
                            break
                        mid = (lo + hi) / 2.0
                        y = np.clip(x + sign * mid * d, -5, 5)
                        fy = call_func(y)
                        if fy < best_hi_fx:
                            best_hi_fx = fy
                            best_hi_step = sign * mid
                            hi = mid
                        else:
                            lo = mid
                    
                    if best_hi_fx < fx:
                        x = x + sign * abs(best_hi_step) * d
                        x = np.clip(x, -5, 5)
                        fx = best_hi_fx
                        improved_in_cycle = True
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                elif best_fx_dir < fx:
                    # Only improvement was from coarse step
                    x = best_x_dir
                    fx = best_fx_dir
                    improved_in_cycle = True
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
            
            # Adapt sigma
            if improved_in_cycle:
                # We made progress, increase sigma slightly to explore further
                sigma = min(sigma * 1.1, sigma_max)
            else:
                # No progress, decrease sigma
                sigma *= 0.5
                if sigma < sigma_min:
                    # Random restart
                    if calls[0] < budget:
                        x_try = rng.uniform(-5, 5, dim)
                        fx_try = call_func(x_try)
                        if fx_try < fx:
                            x = x_try
                            fx = fx_try
                            best_x = x.copy()
                            best_fx = fx
                    sigma = 1.0

        return best_x, best_fx
# EVOLVE-BLOCK-END
