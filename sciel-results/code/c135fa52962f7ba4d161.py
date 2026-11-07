import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def can_call():
            nonlocal calls_made
            if calls_made >= budget:
                return False
            if time.time() > deadline:
                return False
            return True

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if not can_call():
                return float('inf')
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Deterministic Radial Sweep
        # The Discus function f11 is extremely anisotropic. The "needle" (direction of high curvature)
        # is usually aligned with one of the axes or a simple combination.
        # We perform a dense sweep along the 5 axes (positive and negative) to find the needle direction.
        # We also check the origin.
        
        # 1. Evaluate origin
        evaluate(np.zeros(dim))
        
        # 2. Radial sweep along axes
        # We use a logarithmic or dense linear sweep to catch the needle quickly.
        # Since the box is [-5, 5], and the shift is in [-4, 4], the needle is likely within range.
        # We will sample points at distances: 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0
        # along each axis +/-
        
        radii = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        
        for axis in range(dim):
            for r in radii:
                # Positive direction
                x_pos = np.zeros(dim)
                x_pos[axis] = r
                evaluate(x_pos)
                
                # Negative direction
                x_neg = np.zeros(dim)
                x_neg[axis] = -r
                evaluate(x_neg)
                
                if not can_call():
                    break
            if not can_call():
                break
        
        # Phase 2: Fast (1+1)-ES with aggressive adaptation
        # We start from the best point found in the sweep.
        # The (1+1)-ES is very fast and good for anisotropic problems if the step size is managed well.
        # We use a simple adaptive step size that increases on failure and decreases on success.
        # This allows it to "jump" over flat regions or escape local minima if it gets stuck.
        
        x = best_x.copy()
        sigma = 0.5 # Initial step size
        
        # We will run this until we run out of budget or time.
        # We use a large number of iterations, but the loop is very cheap.
        
        while can_call():
            # Generate candidate
            # Use a standard normal distribution
            z = np.random.randn(dim)
            x_trial = x + sigma * z
            
            # Clip to box
            x_trial = np.clip(x_trial, -5, 5)
            
            f_trial = evaluate(x_trial)
            
            if f_trial < best_f:
                # Success
                x = x_trial.copy()
                # Reduce sigma slightly to converge
                sigma *= 0.9
            else:
                # Failure
                # Increase sigma to explore more
                sigma *= 1.1
                
            # Keep sigma within reasonable bounds
            sigma = np.clip(sigma, 1e-6, 10.0)

            # Early stopping if sigma becomes very small and we are close to convergence
            if sigma < 1e-5:
                # Try a few more steps with tiny sigma, then break if no improvement
                # For now, just break to save time for potential restarts if we want, 
                # but with (1+1)-ES, we usually just let it run or break.
                # Given the budget, let's just break if it's really small.
                break

        return best_x
