"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-scale coordinate descent with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit
        
        best_x = None
        best_fx = float('inf')
        calls_made = 0
        
        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        # Phase 1: Coarse grid scan for early progress
        # Use a deterministic grid to ensure coverage
        grid_points = 3
        step_coarse = 10.0 / (grid_points - 1)  # 5.0
        
        for i in range(grid_points):
            if time.time() > deadline:
                break
            x = np.full(self.dim, -5.0 + i * step_coarse)
            fx = func(x)
            calls_made += 1
            update_best(x, fx)
        
        # Phase 2: Multi-scale coordinate descent
        # Start with large steps, shrink geometrically
        scales = [5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125, 0.0390625, 0.01953125, 0.009765625]
        
        # Use the best point from coarse scan as starting point
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_fx = func(best_x)
            calls_made += 1
        
        current_x = best_x.copy()
        current_fx = best_fx
        
        for scale in scales:
            if time.time() > deadline:
                break
            
            # For each dimension, try positive and negative steps
            for dim_idx in range(self.dim):
                if time.time() > deadline:
                    break
                
                # Try positive step
                x_pos = current_x.copy()
                x_pos[dim_idx] += scale
                x_pos = np.clip(x_pos, -5, 5)
                fx_pos = func(x_pos)
                calls_made += 1
                
                if fx_pos < current_fx:
                    current_x = x_pos
                    current_fx = fx_pos
                    update_best(current_x, current_fx)
                
                # Try negative step from original current_x
                x_neg = current_x.copy()
                x_neg[dim_idx] -= scale
                x_neg = np.clip(x_neg, -5, 5)
                fx_neg = func(x_neg)
                calls_made += 1
                
                if fx_neg < current_fx:
                    current_x = x_neg
                    current_fx = fx_neg
                    update_best(current_x, current_fx)
        
        # Phase 3: Fine-tuning with smaller steps if budget remains
        if time.time() < deadline and calls_made < self.budget:
            fine_scales = [0.005, 0.0025, 0.00125, 0.000625]
            
            for scale in fine_scales:
                if time.time() > deadline:
                    break
                
                for dim_idx in range(self.dim):
                    if time.time() > deadline:
                        break
                    
                    x_pos = current_x.copy()
                    x_pos[dim_idx] += scale
                    x_pos = np.clip(x_pos, -5, 5)
                    fx_pos = func(x_pos)
                    calls_made += 1
                    
                    if fx_pos < current_fx:
                        current_x = x_pos
                        current_fx = fx_pos
                        update_best(current_x, current_fx)
                    
                    x_neg = current_x.copy()
                    x_neg[dim_idx] -= scale
                    x_neg = np.clip(x_neg, -5, 5)
                    fx_neg = func(x_neg)
                    calls_made += 1
                    
                    if fx_neg < current_fx:
                        current_x = x_neg
                        current_fx = fx_neg
                        update_best(current_x, current_fx)
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_fx = func(best_x)
            calls_made += 1
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
