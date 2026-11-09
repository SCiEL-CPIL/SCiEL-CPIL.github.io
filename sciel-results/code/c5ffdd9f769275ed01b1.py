"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start Randomized Coordinate Descent for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        calls_made = 0
        best_x = None
        best_f = np.inf
        start_time = time.time()
        
        # Deadline guard: 300s limit, aim for 290s to be safe
        deadline = start_time + 290.0
        
        # Helper to perform a 1D line search along a coordinate
        def line_search(x, axis, step_size, max_iters=10):
            nonlocal calls_made, best_x, best_f
            
            # Initial step in positive direction
            dx = np.zeros(self.dim)
            dx[axis] = step_size
            x_trial = np.clip(x + dx, -5, 5)
            
            # Evaluate initial step
            if calls_made >= self.budget or time.time() > deadline:
                return x, func(x) if calls_made < self.budget else best_f
            
            f_trial = func(x_trial)
            calls_made += 1
            
            if f_trial < best_f:
                best_f = f_trial
                best_x = x_trial.copy()
            
            if f_trial >= func(x):
                # Try negative direction
                dx[axis] = -step_size
                x_trial = np.clip(x + dx, -5, 5)
                if calls_made >= self.budget or time.time() > deadline:
                    return x_trial, f_trial
                f_trial = func(x_trial)
                calls_made += 1
                
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()
                
                if f_trial >= func(x):
                    return x_trial, f_trial # No improvement found
            
            # Golden section search or bisection for finer convergence
            # Since Rastrigin is periodic, simple step halving is robust
            # We do a few steps of refinement
            x_curr = x_trial
            f_curr = f_trial
            
            for _ in range(5):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                
                # Halve the step
                dx_new = (x_curr - x) * 0.5
                x_next = x + dx_new
                x_next = np.clip(x_next, -5, 5)
                
                f_next = func(x_next)
                calls_made += 1
                
                if f_next < f_curr:
                    x, x_curr = x_curr, x_next
                    f_curr = f_next
                    if f_curr < best_f:
                        best_f = f_curr
                        best_x = x_curr.copy()
                else:
                    # Step was too big, but we already moved. 
                    # For simplicity in this budget, we just accept the first improvement 
                    # and let the next restart handle fine tuning, or do one more step back?
                    # Actually, standard RCD moves if it improves. 
                    # Let's just stick to the best found in this coordinate.
                    break
            
            return x_curr, f_curr

        # Multi-start strategy
        # Allocate budget to multiple restarts
        # f4 (Büche-Rastrigin) has many local minima. 
        # Early progress is key for AOCC.
        
        # Start with a few random points
        n_restarts = 50
        budget_per_restart = max(1, self.budget // n_restarts)
        
        # First, do a quick random search to get a decent baseline
        # This ensures we have some calls made and a reasonable best_f early on
        
        # 1. Initial random sampling
        for i in range(min(10, self.budget)):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # 2. Coordinate descent refinement from best point
        if best_x is not None:
            x = best_x.copy()
            f = best_f
            step_size = 0.5
            # Randomize coordinate order
            order = rng.permutation(self.dim)
            for axis in order:
                if calls_made >= self.budget or time.time() > deadline:
                    break
                x, f = line_search(x, axis, step_size)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            # Second pass with smaller step
            if calls_made < self.budget and time.time() < deadline:
                step_size = 0.1
                order = rng.permutation(self.dim)
                for axis in order:
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    x, f = line_search(x, axis, step_size)
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
            
            # Third pass with even smaller step
            if calls_made < self.budget and time.time() < deadline:
                step_size = 0.01
                order = rng.permutation(self.dim)
                for axis in order:
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    x, f = line_search(x, axis, step_size)
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
        
        # 3. Additional restarts to escape local minima
        # Use remaining budget for new random starts and quick CD
        while calls_made < self.budget and time.time() < deadline:
            # Random start near current best or completely random?
            # Completely random is better for multimodal, but might waste calls.
            # Let's mix: 50% random, 50% perturbed best
            if rng.random() < 0.5 and best_x is not None:
                x = best_x + rng.normal(0, 0.5, self.dim)
            else:
                x = rng.uniform(-5, 5, self.dim)
            
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
            
            # Quick coordinate descent on this new point
            # Only if we have budget left
            if calls_made + self.dim * 2 < self.budget:
                step_size = 0.2
                order = rng.permutation(self.dim)
                for axis in order:
                    if calls_made >= self.budget - 5 or time.time() > deadline:
                        break
                    x, f = line_search(x, axis, step_size)
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
