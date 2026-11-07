import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def eval_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or not time_ok():
                return np.inf
            calls += 1
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Coarse grid to identify basin
        # Use 25 evaluations on a structured grid
        grid_budget = 25
        grid_points = []
        # 3^2 grid in 5D is too many, so use a sparse structured sampling
        # Actually, let's use 5 points per dimension in a Latin hypercube-like way
        # Or just use a small set of deterministic points + random
        # Let's use 5^2 = 25 points from a stratified sampling
        
        # Create a stratified sample: divide each dimension into 5 bins
        # and pick one point per combination? That's 5^5 = 3125, too many.
        # Instead, use a smaller structured set.
        
        # Simple approach: use the center, corners, and mid-edges
        # Center
        grid_points.append(np.zeros(dim))
        
        # Corners: 2^5 = 32, too many. Use a subset.
        # Let's use 24 random points + 1 center = 25
        
        for i in range(grid_budget - 1):
            x = rng.uniform(-4.0, 4.0, size=dim)
            grid_points.append(x)
        
        for x in grid_points:
            x = np.clip(x, -5.0, 5.0)
            f = eval_func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if calls >= budget or not time_ok():
            return best_x, best_f

        # Phase 2: (1+1)-ES with adaptive step size
        x = best_x.copy()
        f_x = best_f
        sigma = 0.5  # Initial step size
        
        # Track success rate for adaptive sigma
        success_count = 0
        total_count = 0
        window_size = 10
        
        while calls < budget and time_ok():
            # Generate mutant
            mutation = rng.normal(0, sigma, size=dim)
            x_mut = x + mutation
            x_mut = np.clip(x_mut, -5.0, 5.0)
            
            f_mut = eval_func(x_mut)
            total_count += 1
            
            if f_mut < f_x:
                # Accept mutant
                x = x_mut
                f_x = f_mut
                success_count += 1
                
                # Update global best
                if f_x < best_f:
                    best_f = f_x
                    best_x = x.copy()
                
                # Adaptive sigma: increase if success rate is high
                if total_count % window_size == 0:
                    success_rate = success_count / window_size
                    if success_rate > 0.5:
                        sigma *= 1.1
                    elif success_rate < 0.2:
                        sigma *= 0.8
                    success_count = 0
            else:
                # Reject mutant
                if total_count % window_size == 0:
                    success_rate = success_count / window_size
                    if success_rate < 0.2:
                        sigma *= 0.8
                    success_count = 0
        
        return best_x, best_f
