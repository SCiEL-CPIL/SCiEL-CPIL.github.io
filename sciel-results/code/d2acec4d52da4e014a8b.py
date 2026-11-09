"""Portfolio search for BBOB f7 (Step ellipsoidal) with ILS and Coordinate Descent."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ILS with Coordinate Descent refinement for Step Ellipsoidal."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Initialize
        best_x = rng.uniform(-5.0, 5.0, dim)
        best_f = func(best_x)
        calls = 1

        # Estimate plateau level
        # Sample a few more points to be sure we have a good baseline for the plateau
        # If we find a lower value immediately, great.
        plateau_val = best_f
        for _ in range(10):
            if calls >= budget or time.time() > deadline:
                break
            x_sample = rng.uniform(-5.0, 5.0, dim)
            f_sample = func(x_sample)
            calls += 1
            if f_sample < best_f:
                best_f = f_sample
                best_x = x_sample.copy()
            else:
                # If f_sample is close to plateau_val, update plateau_val
                # Usually plateau is the max or a constant high value
                if abs(f_sample - plateau_val) < 1e-8:
                    # consistent plateau
                    pass
                elif f_sample > plateau_val:
                    plateau_val = f_sample

        # If we haven't found anything better than the first sample, 
        # it's likely we are on the plateau.
        # We need to find the ellipsoid.

        # Local Refinement Function: Coordinate Descent
        def refine(x_start, f_start, remaining_budget, max_iter=50):
            nonlocal calls
            x = x_start.copy()
            f_curr = f_start
            step = 1.0
            
            for _ in range(max_iter):
                if calls >= remaining_budget or time.time() > deadline:
                    break
                improved = False
                for i in range(dim):
                    if calls >= remaining_budget or time.time() > deadline:
                        break
                    # Try positive direction
                    x_trial = x.copy()
                    x_trial[i] += step
                    x_trial = np.clip(x_trial, -5.0, 5.0)
                    f_trial = func(x_trial)
                    calls += 1
                    if f_trial < f_curr:
                        x = x_trial
                        f_curr = f_trial
                        improved = True
                    # Try negative direction
                    if calls >= remaining_budget or time.time() > deadline:
                        break
                    x_trial = x.copy()
                    x_trial[i] -= 2 * step  # Move from original x? No, from current x?
                    # Standard coordinate descent: evaluate x +/- step from current x
                    # Let's restart from x before this dim change?
                    # Simpler: 
                    # x_plus = x.copy(); x_plus[i] += step
                    # x_minus = x.copy(); x_minus[i] -= step
                    # Pick the better one.
                    
                    # Let's redo the logic inside the loop to be cleaner
                    pass 
                if not improved:
                    step *= 0.5
                    if step < 1e-5:
                        break
            return x, f_curr

        # The refine function above is a bit messy with the "nonlocal" and loop structure.
        # Let's write a cleaner, inline version or a proper helper.
        
        def coordinate_descent(x_start, f_start, budget_limit):
            x = x_start.copy()
            f_curr = f_start
            step = 1.0
            max_iter = 40 # Cap iterations to save budget for global search
            
            for _ in range(max_iter):
                if calls >= budget_limit or time.time() > deadline:
                    break
                improved = False
                for i in range(dim):
                    if calls >= budget_limit or time.time() > deadline:
                        break
                    
                    # Evaluate + step
                    x_plus = x.copy()
                    x_plus[i] += step
                    x_plus = np.clip(x_plus, -5.0, 5.0)
                    f_plus = func(x_plus)
                    calls += 1
                    
                    # Evaluate - step
                    x_minus = x.copy()
                    x_minus[i] -= step
                    x_minus = np.clip(x_minus, -5.0, 5.0)
                    f_minus = func(x_minus)
                    calls += 1
                    
                    # Pick best
                    if f_plus < f_minus and f_plus < f_curr:
                        x = x_plus
                        f_curr = f_plus
                        improved = True
                    elif f_minus < f_curr:
                        x = x_minus
                        f_curr = f_minus
                        improved = True
                        
                if not improved:
                    step *= 0.7
                    if step < 1e-6:
                        break
            return x, f_curr

        # Main Loop: ILS
        # Strategy: 
        # 1. Jump to a random point.
        # 2. If it's better than best_f, or if it's "inside" (f < plateau_val), refine it.
        # 3. Update global best.
        
        # We already did some sampling. 
        # Let's iterate.
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate a new candidate
            # Mix of large jumps and smaller local perturbations if we are doing well?
            # For f7, large jumps are needed to hit the ellipsoid.
            x_candidate = rng.uniform(-5.0, 5.0, dim)
            f_candidate = func(x_candidate)
            calls += 1
            
            # Check if we found something interesting
            # "Interesting" means f_candidate < plateau_val (inside ellipsoid)
            # Or f_candidate < best_f
            if f_candidate < plateau_val - 1e-8 or f_candidate < best_f:
                # Refine this candidate
                x_refined, f_refined = coordinate_descent(x_candidate, f_candidate, budget)
                
                if f_refined < best_f:
                    best_f = f_refined
                    best_x = x_refined.copy()
                    
                # If we are inside the ellipsoid, we might want to do more local search?
                # Coordinate descent should be close to the minimum.
                # We can continue the ILS loop to find other potential minima (though f7 has 1).
                # But maybe the coordinate descent didn't fully converge?
                # We can run a second pass with smaller initial step?
                # Or just rely on the step reduction in coordinate_descent.
                
                # If we are still on the plateau (f_refined == plateau_val), 
                # it means we jumped to the plateau, not the ellipsoid.
                # In that case, we just continue.

        return best_x, best_f
# EVOLVE-BLOCK-END
