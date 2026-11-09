import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 330s limit
        
        best_x = None
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(42)
        
        def wrapped_func(x):
            nonlocal best_x, best_f, call_count
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if call_count >= self.budget:
                raise StopIteration("Budget exceeded")
            
            fx = func(x)
            call_count += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            return fx

        # Phase 1: Coarse grid search to find promising regions
        # Use a 3^dim grid in [-4, 4]^dim (since minimizer is in [-4, 4]^5)
        # 3^5 = 243 points, affordable with budget 500
        grid_size = 3
        grid_points = []
        for i in range(grid_size ** self.dim):
            coords = []
            temp = i
            for d in range(self.dim):
                coords.append(temp % grid_size)
                temp //= grid_size
            x = np.array([-4.0 + 8.0 * c / (grid_size - 1) for c in coords])
            grid_points.append(x)
        
        # Evaluate grid points (shuffled to spread out)
        np.random.seed(42)
        np.random.shuffle(grid_points)
        
        for x in grid_points:
            if time.time() > deadline or call_count >= self.budget:
                break
            try:
                wrapped_func(x)
            except (TimeoutError, StopIteration):
                break
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            call_count = 1
        
        # Phase 2: Local refinement using CMA-ES-like strategy
        # Use multiple restarts from best point with decreasing step sizes
        remaining_budget = self.budget - call_count
        if remaining_budget > 0 and best_x is not None:
            # Try several local searches from best point
            num_restarts = min(5, max(1, remaining_budget // 50))
            step_sizes = [1.0, 0.5, 0.2, 0.1, 0.05][:num_restarts]
            
            for sigma in step_sizes:
                if call_count >= self.budget or time.time() > deadline:
                    break
                
                # Start from best point with some perturbation
                x = best_x.copy() + sigma * rng.standard_normal(self.dim)
                x = np.clip(x, -5.0, 5.0)
                
                # Simple (1+1)-ES from this point
                local_best_x = x.copy()
                local_best_f = wrapped_func(x)
                local_sigma = sigma
                
                for _ in range(min(remaining_budget - (call_count - (self.budget - remaining_budget)), 80)):
                    if call_count >= self.budget or time.time() > deadline:
                        break
                    
                    y = np.clip(local_best_x + local_sigma * rng.standard_normal(self.dim), -5.0, 5.0)
                    fy = wrapped_func(y)
                    
                    if fy < local_best_f:
                        local_best_x = y.copy()
                        local_best_f = fy
                        local_sigma *= 1.3
                    else:
                        local_sigma *= 0.85
                    
                    # Update global best
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()
                
                remaining_budget = self.budget - call_count
                if remaining_budget <= 0:
                    break
        
        return best_x, best_f
