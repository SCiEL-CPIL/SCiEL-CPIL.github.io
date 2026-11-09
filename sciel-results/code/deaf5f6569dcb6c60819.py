"""Deterministic Multi-Resolution Grid Descent for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Deterministic Multi-Resolution Grid Descent optimizer for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit. 280s is a safe margin to ensure we return before kill.
        deadline = start_time + 280.0
        
        d = self.dim
        budget = self.budget
        
        # Global best tracking
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        # Helper to evaluate a point
        def eval_point(x):
            nonlocal best_f, best_x, calls_used
            if calls_used >= budget or time.time() > deadline:
                return None
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse Global Search
        # We need to find a good starting point. Since the minimum is in [-4, 4]^5,
        # we can search this region.
        # A full grid of 3^5 = 243 points is cheap.
        # Let's do a coarse grid over [-4, 4]^5 with step 2.0.
        # Points: -4, -2, 0, 2, 4 -> 5 points per dim -> 5^5 = 3125. Too many.
        # Let's use step 4.0: -4, 0, 4 -> 3 points per dim -> 3^5 = 243 points.
        # This covers the likely region of the minimum.
        
        # Actually, the contract says minimizer lies in [-4, 4]^5.
        # Let's do a 3-point grid per dimension in [-4, 4].
        grid_vals = np.array([-4.0, 0.0, 4.0])
        
        # Generate all combinations
        coords = np.array(np.meshgrid(*[grid_vals]*d)).T.reshape(-1, d)
        
        for x in coords:
            eval_point(x)
            
        if best_x is None:
            # Fallback if something went wrong
            x = np.zeros(d)
            eval_point(x)
            if best_x is None:
                return np.zeros(d), np.inf

        current_best = best_x.copy()
        
        # Phase 2: Multi-Resolution Local Refinement
        # We will perform a series of grid searches centered on the current best.
        # The resolution (step size) decreases in each phase.
        # The search radius also decreases.
        
        # Step sizes for the grid:
        # Phase 2: Step 1.0, Radius 2.0 (covers [-2, 2] around best)
        # Phase 3: Step 0.5, Radius 1.0
        # Phase 4: Step 0.1, Radius 0.5
        # Phase 5: Step 0.02, Radius 0.2
        # Phase 6: Step 0.005, Radius 0.1
        # Phase 7: Step 0.001, Radius 0.05
        
        # We need to be careful with the number of points.
        # For a grid of step size h in a box of size R in d dimensions:
        # Number of points = (2*R/h + 1)^d
        # We must check if we have budget left.
        
        phases = [
            (1.0, 2.0),
            (0.5, 1.0),
            (0.1, 0.5),
            (0.02, 0.2),
            (0.005, 0.1),
            (0.001, 0.05),
            (0.0005, 0.02),
            (0.0001, 0.01)
        ]
        
        for step, radius in phases:
            if calls_used >= budget or time.time() > deadline:
                break
                
            # Define grid points
            # We want to search current_best +/- radius
            # Number of points per dim: n = int(2 * radius / step)
            # To ensure symmetric coverage, let's use odd number of points.
            n = int(2 * radius / step)
            if n % 2 == 0:
                n += 1
            n = max(n, 3) # At least 3 points (-1, 0, 1)
            
            # Total points
            total_points = n ** d
            
            # Check budget
            if total_points > (budget - calls_used):
                # If we don't have enough budget for the full grid,
                # we can either skip this phase or reduce n.
                # Let's try to reduce n to fit.
                max_points = budget - calls_used
                if max_points <= 0:
                    break
                
                # Find largest n such that n^d <= max_points
                # n <= max_points^(1/d)
                n_max = int(max_points ** (1.0 / d))
                if n_max < 3:
                    break
                # Make it odd
                if n_max % 2 == 0:
                    n_max -= 1
                n = n_max
                
                # Recalculate step to maintain radius? Or just use fewer points?
                # If we reduce n, the effective step increases.
                # Let's just use the calculated n.
                pass
            
            # Generate offsets
            offsets = np.linspace(-radius, radius, n)
            coords = np.array(np.meshgrid(*[offsets]*d)).T.reshape(-1, d)
            
            # Evaluate
            for offset in coords:
                if calls_used >= budget or time.time() > deadline:
                    break
                x_new = current_best + offset
                # Clip to domain
                x_new = np.clip(x_new, -5.0, 5.0)
                
                fx = func(x_new)
                calls_used += 1
                
                if fx < best_f:
                    best_f = fx
                    best_x = x_new.copy()
                    # If we found a better point, we might want to restart the grid?
                    # For simplicity, we just update best_x.
                    # The next phase will center on the new best_x.
                    # However, within this phase, we continue with the original center.
                    # This is a valid strategy.
            
            # Update center for next phase
            current_best = best_x.copy()
            
            # If we are making very little progress, we might want to stop early?
            # Or just let it run. The grid is deterministic and covers the area.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
