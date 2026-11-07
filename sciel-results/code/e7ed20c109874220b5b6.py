import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270.0
        
        best_x = np.zeros(n)
        best_f = float('inf')
        evals_left = budget
        
        # Helper to generate grid points
        def get_grid_points(low, high, subs):
            indices = np.arange(subs)
            grids = np.meshgrid(*([indices] * n), indexing='ij')
            grid_indices = np.stack(grids, axis=-1).reshape(-1, n)
            cell_width = (high - low) / subs
            centers = low + (grid_indices + 0.5) * cell_width
            return centers

        # Phase 1: Coarse Grid
        low = np.full(n, -5.0)
        high = np.full(n, 5.0)
        
        # 2 subdivisions -> 32 points
        subs1 = 2
        points1 = get_grid_points(low, high, subs1)
        
        if evals_left < len(points1):
            # Not enough budget, just do random search or partial grid
            # But budget is 500, so this shouldn't happen
            pass

        # Evaluate Phase 1
        for p in points1:
            if evals_left <= 0 or time.time() > deadline:
                break
            f = func(p)
            if f < best_f:
                best_f = f
                best_x = p.copy()
            evals_left -= 1

        # Identify best cell for Phase 1
        cell_width1 = (high - low) / subs1
        # Find cell indices for best_x
        # Clip best_x to strictly inside to handle boundaries
        bx_clipped = np.clip(best_x, low + 1e-9, high - 1e-9)
        cell_idx1 = np.floor((bx_clipped - low) / cell_width1).astype(int)
        cell_idx1 = np.clip(cell_idx1, 0, subs1 - 1)
        
        best_low1 = low + cell_idx1 * cell_width1
        best_high1 = best_low1 + cell_width1

        # Phase 2: Medium Grid in best cell
        subs2 = 3 # 3^5 = 243 points
        points2 = get_grid_points(best_low1, best_high1, subs2)
        
        for p in points2:
            if evals_left <= 0 or time.time() > deadline:
                break
            f = func(p)
            if f < best_f:
                best_f = f
                best_x = p.copy()
            evals_left -= 1

        # Phase 3: Local Search with (1+1)-ES
        # Use remaining budget
        x = best_x.copy()
        fx = best_f
        sigma = 0.1 # Small initial step
        
        while evals_left > 0 and time.time() < deadline:
            # Simple (1+1)-ES
            y = x + sigma * np.random.standard_normal(n)
            y = np.clip(y, -5, 5)
            fy = func(y)
            evals_left -= 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1 # Slight increase
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.9 # Decrease
                # If sigma is too small, break
                if sigma < 1e-6:
                    break

        return best_x, best_f
