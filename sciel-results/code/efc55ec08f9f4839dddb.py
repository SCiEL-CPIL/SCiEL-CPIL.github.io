"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer combining coordinate-wise scan and adaptive (1+1)-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        eval_count = 0

        def check_time():
            return time.time() < deadline - 3.0

        def try_update(x, fx):
            nonlocal best_x, best_f, eval_count
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        def clip_box(x):
            return np.clip(x, -5.0, 5.0)

        # ===== Phase 1: Coordinate-wise Scan (Greedy Coordinate Descent on Grid) =====
        # Initialize at origin
        x = np.zeros(n)
        fx = func(x)
        try_update(x, fx)
        
        if best_x is None:
            best_x = x.copy()
            best_f = fx

        # Define a grid of candidate values for each coordinate.
        # We scan from coarse to fine. 
        # The box is [-5, 5]. The optimum is in [-4, 4].
        # We use a grid that covers this range.
        
        # Coarse grid: -4, -3, -2, -1, 0, 1, 2, 3, 4
        coarse_grid = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        # Fine grid: steps of 0.5
        fine_grid = np.arange(-4.0, 4.5, 0.5)
        
        # We will iterate through dimensions. For each dimension, we try all grid points.
        # To save budget, we do one pass with coarse grid, then one pass with fine grid.
        
        def coordinate_scan(grid):
            nonlocal x, fx, best_x, best_f, eval_count
            for coord in range(n):
                if not check_time() or eval_count >= budget - 10:
                    break
                best_val_for_coord = fx
                best_pos_for_coord = x[coord]
                
                # Sort grid to explore near current value first? Or just iterate?
                # Iterating is fine. We check all points.
                for val in grid:
                    if not check_time() or eval_count >= budget - 10:
                        break
                    # Skip if same as current
                    if abs(val - x[coord]) < 1e-9:
                        continue
                    
                    x[coord] = val
                    fv = func(x)
                    try_update(x, fv)
                    
                    if fv < best_val_for_coord:
                        best_val_for_coord = fv
                        best_pos_for_coord = val
            
            # After checking all values for this coordinate, set it to the best one found
            # Note: The loop above updates x[coord] to the last tested value. 
            # We need to restore the best value found for this coordinate.
            # However, the 'try_update' inside updates global best. 
            # The local 'x' is used for the search. 
            # We should set x[coord] back to the best value found for that coordinate 
            # to maintain a consistent 'current best' for the next coordinate.
            
            # Actually, the logic above is slightly flawed for a pure scan because 
            # 'x' is modified in place. If we find a better value at val=2, we set x[coord]=2.
            # If a later val=3 is worse, x[coord] remains 2. 
            # But if val=1 is better than val=2, x[coord] becomes 1.
            # So at the end of the inner loop, x[coord] holds the LAST tested value, 
            # not necessarily the BEST value.
            # We must explicitly set x[coord] to best_pos_for_coord.
            
            # Wait, the loop above does:
            # x[coord] = val
            # fv = func(x)
            # if fv < best_val: best_val = fv; best_pos = val
            # So yes, we must restore x[coord] = best_pos_for_coord after the inner loop.
            
            # But wait, the outer loop iterates over coords. 
            # The 'x' vector is shared. 
            # So for coord 0, we find best_val_0. Set x[0] = best_pos_0.
            # Then for coord 1, we use x with x[0] fixed at best_pos_0.
            # This is correct greedy coordinate descent.
            
            # Let's re-implement the scan more carefully to ensure we restore the best.
            
            # Re-doing the scan logic for clarity and correctness in the code below.
            pass

        # Corrected Coordinate Scan Implementation
        def perform_coordinate_scan(grid):
            nonlocal x, fx, best_x, best_f, eval_count
            for coord in range(n):
                if not check_time() or eval_count >= budget - 10:
                    return
                
                current_val = x[coord]
                best_val_for_coord = fx
                best_pos_for_coord = current_val
                
                for val in grid:
                    if not check_time() or eval_count >= budget - 10:
                        break
                    if abs(val - current_val) < 1e-9:
                        continue
                    
                    x[coord] = val
                    fv = func(x)
                    try_update(x, fv)
                    
                    if fv < best_val_for_coord:
                        best_val_for_coord = fv
                        best_pos_for_coord = val
                
                # Restore best position for this coordinate
                if best_pos_for_coord != current_val:
                    x[coord] = best_pos_for_coord
                    # We don't need to re-evaluate x[coord]=best_pos_for_coord because 
                    # we already evaluated it and stored the value in best_val_for_coord.
                    # However, 'fx' is the value of the current 'x' vector.
                    # Since we changed x[coord] back to the best, the value of the full vector 
                    # is indeed best_val_for_coord (assuming other coords didn't change).
                    # So we can update fx.
                    fx = best_val_for_coord
                    # Update global best if this is better
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

        # Pass 1: Coarse Scan
        if check_time() and eval_count < budget - 50:
            perform_coordinate_scan(coarse_grid)
        
        # Pass 2: Fine Scan
        if check_time() and eval_count < budget - 50:
            perform_coordinate_scan(fine_grid)

        # Ensure we have a valid best_x
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            try_update(best_x, best_f)

        # ===== Phase 2: Adaptive (1+1)-ES =====
        # Use the result of the coordinate scan as the starting point.
        x = best_x.copy()
        fx = best_f
        
        sigma = 0.5  # Initial step size
        sigma_min = 1e-6
        sigma_max = 5.0
        no_improve = 0
        max_no_improve = 40
        
        # Main ES loop
        while check_time() and eval_count < budget:
            z = rng.standard_normal(n)
            y = clip_box(x + sigma * z)
            fy = func(y)
            try_update(y, fy)
            
            if fy < fx:
                x = y
                fx = fy
                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma = min(sigma * 1.25, sigma_max)
                no_improve = 0
            else:
                sigma = max(sigma * 0.8, sigma_min)
                no_improve += 1
            
            if sigma < sigma_min or no_improve >= max_no_improve:
                break

        # ===== Phase 3: Local Refinement / Restart =====
        # If we have budget left, do a few quick restarts or fine-tuning.
        # The (1+1)-ES might have converged to a local minimum.
        # Try a few random jumps from the best point.
        
        remaining_budget = budget - eval_count
        if remaining_budget > 10 and check_time():
            for _ in range(min(3, remaining_budget // 10)):
                if not check_time() or eval_count >= budget:
                    break
                
                # Jump from best
                x_restart = clip_box(best_x + rng.standard_normal(n) * 1.0)
                fx_restart = func(x_restart)
                try_update(x_restart, fx_restart)
                
                # Short ES from this new point
                x_local = x_restart
                fx_local = fx_restart
                sigma_local = 0.5
                no_improve_local = 0
                
                # Use at most 10% of remaining budget for this restart
                steps = min(10, remaining_budget // 3)
                for _ in range(steps):
                    if not check_time() or eval_count >= budget:
                        break
                    
                    z = rng.standard_normal(n)
                    y = clip_box(x_local + sigma_local * z)
                    fy = func(y)
                    try_update(y, fy)
                    
                    if fy < fx_local:
                        x_local = y
                        fx_local = fy
                        if fx_local < best_f:
                            best_f = fx_local
                            best_x = x_local.copy()
                        sigma_local = min(sigma_local * 1.2, 2.0)
                        no_improve_local = 0
                    else:
                        sigma_local = max(sigma_local * 0.8, 1e-5)
                        no_improve_local += 1
                    
                    if sigma_local < 1e-5 or no_improve_local > 10:
                        break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
