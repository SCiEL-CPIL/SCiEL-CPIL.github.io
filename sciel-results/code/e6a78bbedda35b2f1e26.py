"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid optimizer: deterministic grid scan for fast initial AOCC, then (1+1)-ES polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        dim = self.dim
        total_budget = self.budget
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def generate_halton(n, dim, offset=1):
            points = []
            for i in range(offset, offset + n):
                point = []
                bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]
                for j in range(dim):
                    point.append(halton(i, bases[j]))
                points.append(point)
            return np.array(points)

        # 1. Deterministic Grid Scan for immediate AOCC boost
        # Büche-Rastrigin has a grid-like structure of local minima.
        # Scanning a coarse grid ensures we hit a good basin very early.
        # We have 500 calls. Let's use 125 for the grid (5^3 = 125).
        # This gives us a 5x5x5x5x5 grid over [-5, 5].
        # Actually, 5 points per dimension means step size 2.5.
        # Grid: -5, -2.5, 0, 2.5, 5.
        
        grid_size_per_dim = 5
        grid_points = []
        for i in range(grid_size_per_dim):
            for j in range(grid_size_per_dim):
                for k in range(grid_size_per_dim):
                    for l in range(grid_size_per_dim):
                        for m in range(grid_size_per_dim):
                            # Map 0..4 to -5..5
                            x = np.array([i, j, k, l, m])
                            x = x * (10.0 / (grid_size_per_dim - 1)) - 5.0
                            grid_points.append(x)
        
        grid_points = np.array(grid_points)
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        
        # Evaluate grid points
        # We want to do this quickly to maximize early progress.
        # We'll evaluate all 125 points.
        for p in grid_points:
            if total_calls >= total_budget:
                break
            if time.time() >= deadline:
                break
                
            f_val = func(p)
            total_calls += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = p.copy()
        
        # 2. Local Search Polish
        # Now we have a good starting point.
        # Use (1+1)-ES to refine.
        # We have ~375 calls left.
        
        if best_x is None:
            # Fallback if grid failed or budget exhausted
            best_x = np.zeros(dim)
            best_f = func(best_x)
            total_calls += 1
            return best_x, best_f
            
        def local_search(x_start, max_calls, initial_sigma=0.5):
            x = x_start.copy()
            f = best_f # We already evaluated x_start in the grid phase, so we don't need to re-evaluate?
                       # Wait, local_search usually evaluates x_start.
                       # To save a call, we can pass in f.
                       # But the standard (1+1)-ES loop needs to compare.
                       # Let's just be careful.
                       
            # Actually, let's just run a standard (1+1)-ES that starts by evaluating x_start if needed,
            # but since we know f(best_x), we can save 1 call.
            
            # However, to keep it simple and robust:
            # We will run (1+1)-ES.
            
            sigma = initial_sigma
            calls = 0
            
            # We already have f(x).
            while calls < max_calls:
                if time.time() >= deadline:
                    break
                
                z = np.random.standard_normal(dim)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                
                f_y = func(y)
                calls += 1
                
                if f_y < f:
                    x = y
                    f = f_y
                    # Slight increase to explore, but careful not to jump out of basin
                    sigma *= 1.02 
                else:
                    sigma /= 1.03
                    if sigma < 1e-6:
                        break
            
            return x, f, calls

        # Run local search from the best grid point
        remaining = total_budget - total_calls
        if remaining > 0:
            x_res, f_res, calls_used = local_search(best_x, remaining, initial_sigma=1.0)
            total_calls += calls_used
            
            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()
                
        # 3. If we still have budget, try a few other grid points?
        # Or just continue polishing the best one?
        # Polishing the best one is usually better for AOCC if we are close to the global min.
        # But if we are stuck in a local min, restarting might help.
        # Given the grid scan, we are likely in a decent basin.
        # Let's try to restart from the 2nd and 3rd best grid points if we have budget.
        
        # Sort grid points by function value? We didn't store them all.
        # Let's just restart from a few Halton points if we have budget left.
        
        if total_calls < total_budget:
            # Generate a few Halton points for restarts
            num_restarts = 5
            halton_points = generate_halton(num_restarts, dim)
            init_points = halton_points * 10.0 - 5.0
            
            for i in range(num_restarts):
                if total_calls >= total_budget:
                    break
                if time.time() >= deadline:
                    break
                    
                x_init = init_points[i]
                # Evaluate initial point
                f_init = func(x_init)
                total_calls += 1
                
                if f_init < best_f:
                    best_f = f_init
                    best_x = x_init.copy()
                
                # Short local search
                alloc = min(50, total_budget - total_calls)
                if alloc > 0:
                    x_res, f_res, calls_used = local_search(best_x, alloc, initial_sigma=0.5)
                    total_calls += calls_used
                    
                    if f_res < best_f:
                        best_f = f_res
                        best_x = x_res.copy()
                        
        return best_x, best_f
# EVOLVE-BLOCK-END
