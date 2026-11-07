import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0  # Safety margin against 330s limit
        
        def time_check():
            return (time.time() - start_time) > timeout

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        eval_count = 0
        
        # Strategy: Adaptive Coordinate Descent with Binary Search
        # 1. Global Exploration: Randomly sample a few points to find a good starting neighborhood.
        #    The step ellipsoidal function has a "plateau" around the minimum.
        #    We need to get close enough to the minimum so that the step boundaries are within our search range.
        
        n_global = 20
        
        for _ in range(n_global):
            if eval_count >= budget or time_check():
                break
            x = rng.uniform(-5.0, 5.0, dim)
            f = func(x)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
        if eval_count >= budget or time_check():
            return best_x, best_f
            
        # 2. Local Refinement: Coordinate Descent with Binary Search
        # For each dimension, we perform a binary search to find the step boundary.
        # The function f7 is defined as:
        # f(x) = 0.5*||x||^2 + ||x||^2 * indicator(||x|| > 1)
        # Wait, BBOB f7 is "Step Ellipsoidal".
        # f(x) = 0.5 * ||x||^2 + ||x||^2 * I(||x|| > 1)
        # Actually, the standard BBOB f7 is:
        # f(x) = 0.5 * ||x||^2 + ||x||^2 * I(||x|| > 1)
        # This creates a step in the gradient at ||x||=1.
        # The minimum is at the origin (shifted/rotated).
        # The "step" is a discontinuity in the derivative, but the function itself is continuous.
        # However, the prompt says "function values change discontinuously".
        # Let's check the definition of BBOB f7.
        # BBOB f7: Step Ellipsoidal
        # f(x) = 0.5 * ||x||^2 + ||x||^2 * I(||x|| > 1)
        # This is continuous at ||x||=1. f(1) = 0.5 + 1 = 1.5. f(1-eps) = 0.5.
        # Wait, if I(||x||>1) is 0 for ||x||<=1 and 1 for ||x||>1.
        # At ||x||=1, f(x) = 0.5.
        # At ||x||=1+eps, f(x) = 0.5*(1+eps)^2 + (1+eps)^2 ~ 1.5.
        # So there IS a jump in the function value if the indicator is defined strictly.
        # Usually, BBOB functions are continuous.
        # Let's re-read the prompt: "function values change discontinuously rather than smoothly".
        # This implies the specific instance might have a hard step.
        # Regardless, binary search on the axis is a good strategy for step-like functions.
        
        # We will iterate through dimensions multiple times.
        # For each dimension, we define a search interval [lo, hi] around the current best_x.
        # We use binary search to find the point where the function value drops.
        
        # Initial interval size: 2.0 (covering [-1, 1] relative to current best)
        # If the step is within this range, binary search will find it.
        # If not, we might need to expand or shift.
        # Since we did a global search, we assume we are reasonably close.
        
        # Let's do 10 rounds of coordinate descent.
        # Each round: for each dim, do a binary search with ~20 iterations.
        # Total evals: 10 * 5 * 20 = 1000. This exceeds budget.
        # We have ~480 evals left.
        # Let's do 4 rounds. 4 * 5 * 20 = 400. That fits.
        
        max_rounds = 4
        iters_per_search = 20
        
        for _round in range(max_rounds):
            if eval_count >= budget or time_check():
                break
                
            for d in range(dim):
                if eval_count >= budget or time_check():
                    break
                    
                # Define search range for dimension d
                # Center at best_x[d]
                # Width 2.0 (from -1 to +1 relative to center)
                center = best_x[d]
                lo = center - 1.0
                hi = center + 1.0
                
                # Ensure bounds are within [-5, 5]
                lo = max(-5.0, lo)
                hi = min(5.0, hi)
                
                # Binary search for the minimum along this axis
                # We evaluate at mid points.
                # If f(mid) < f(best_x), we move best_x towards mid.
                # If f(mid) > f(best_x), we move away.
                # This is a standard 1D search.
                
                # We need to compare against the current best_f.
                # But note: changing x[d] might make the function worse if we move away from the minimum.
                # So we only update best_x if we find a strictly better point.
                
                # To make it efficient, we can do a ternary search or binary search on the derivative?
                # No, we don't have derivative.
                # Binary search on the "step" boundary:
                # If the function is flat on one side and higher on the other, we can find the boundary.
                # But we don't know which side is the minimum.
                # So we do a simple line search:
                # Evaluate at several points in [lo, hi] and pick the best.
                # Or, better: use a golden section search or similar.
                # But binary search on the step is more efficient if we know the structure.
                # Let's stick to a robust 1D minimization using a small number of evaluations.
                
                # Let's use a simple grid search within the interval for the first few iterations,
                # then refine.
                # Or just do 20 binary search steps assuming unimodality?
                # The step ellipsoidal function is unimodal along any axis passing through the minimum.
                # If our interval contains the minimum, binary search (or golden section) will work.
                # If not, we might converge to the boundary.
                
                # Let's use a modified binary search that keeps track of the best point found so far in this 1D search.
                
                x_test = best_x.copy()
                
                # We'll do a simple "hill climbing" with step reduction.
                # Step size starts at 1.0 and halves each iteration.
                step = 1.0
                current_x = best_x.copy()
                current_f = best_f
                
                for _iter in range(iters_per_search):
                    if eval_count >= budget or time_check():
                        break
                    
                    # Try moving in positive direction
                    x_pos = current_x.copy()
                    x_pos[d] += step
                    if x_pos[d] > 5.0:
                        x_pos[d] = 5.0
                    f_pos = func(x_pos)
                    eval_count += 1
                    
                    if f_pos < current_f:
                        current_x = x_pos
                        current_f = f_pos
                        # Continue in same direction
                    else:
                        # Try moving in negative direction
                        x_neg = current_x.copy()
                        x_neg[d] -= step
                        if x_neg[d] < -5.0:
                            x_neg[d] = -5.0
                        f_neg = func(x_neg)
                        eval_count += 1
                        
                        if f_neg < current_f:
                            current_x = x_neg
                            current_f = f_neg
                        else:
                            # No improvement in either direction, reduce step size
                            step *= 0.5
                            if step < 1e-5:
                                break
                    
                    # Update global best if this local search found something better
                    if current_f < best_f:
                        best_f = current_f
                        best_x = current_x.copy()
                        
        return best_x, best_f
