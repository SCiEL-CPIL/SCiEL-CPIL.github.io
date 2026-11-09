import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin
        
        x_best = None
        f_best = np.inf
        calls_made = 0
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Global Search via Random Sampling
        n_initial = min(50, self.budget // 10)
        if n_initial < 5:
            n_initial = 5
        n_initial = min(n_initial, self.budget)
        
        for i in range(n_initial):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            f = func(x)
            calls_made += 1
            if f < f_best:
                f_best = f
                x_best = x.copy()
                
        if x_best is None:
            x_best = np.zeros(self.dim)
            f_best = func(x_best)
            calls_made += 1
        
        # Phase 2: Powell's Method for local refinement
        # Powell's method maintains a set of conjugate directions
        # and performs line searches along each direction
        
        def powell_method(x_start, f_start, max_calls):
            nonlocal calls_made, f_best, x_best
            
            n = self.dim
            # Initialize directions as unit vectors
            directions = np.eye(n)
            x = x_start.copy()
            f = f_start
            
            # Store best point found during line searches
            x_best_local = x.copy()
            f_best_local = f
            
            # We'll do iterations of Powell's method
            # Each iteration: line search along each direction, then check if new direction is needed
            # A line search uses a few function evaluations
            # Budget per line search: we'll use a simple golden section or parabolic search
            # But to save budget, we'll use a simpler approach: try a few steps
            
            # For each iteration, we do n line searches (one per direction)
            # Plus potentially one extra for the new conjugate direction
            # So each "full iteration" costs about n+1 line searches
            # Each line search costs ~5-10 function evaluations
            # So a full iteration costs ~5n to 10n evaluations
            # With budget of ~450 left and n=5, we can do ~9-18 full iterations
            
            # We'll limit the number of full iterations
            max_iterations = max(1, (max_calls - 10) // (n * 8))  # Conservative estimate
            
            for iteration in range(max_iterations):
                if time.time() > deadline:
                    break
                if calls_made >= self.budget:
                    break
                
                x_old = x.copy()
                f_old = f
                x_min = x.copy()
                f_min = f
                dir_min = 0
                
                # Line search along each direction
                for i in range(n):
                    if time.time() > deadline or calls_made >= self.budget:
                        break
                    
                    # Perform line search along directions[i]
                    # Simple approach: try a few steps
                    # Start with a small step and adjust
                    step = 0.5
                    x_trial = x.copy()
                    f_trial = f
                    
                    # Try to find a better point along this direction
                    # We'll do a simple pattern: try positive and negative steps
                    best_x_dir = x.copy()
                    best_f_dir = f
                    
                    # Try a few step sizes
                    for step_size in [0.5, 0.25, 0.1, 0.05, 0.01]:
                        if time.time() > deadline or calls_made >= self.budget:
                            break
                        
                        # Try positive step
                        x_pos = x + step_size * directions[i]
                        x_pos = np.clip(x_pos, -5, 5)
                        if not np.allclose(x_pos, x):  # Skip if no change
                            f_pos = func(x_pos)
                            calls_made += 1
                            if f_pos < best_f_dir:
                                best_f_dir = f_pos
                                best_x_dir = x_pos.copy()
                        
                        # Try negative step
                        x_neg = x - step_size * directions[i]
                        x_neg = np.clip(x_neg, -5, 5)
                        if not np.allclose(x_neg, x):
                            f_neg = func(x_neg)
                            calls_made += 1
                            if f_neg < best_f_dir:
                                best_f_dir = f_neg
                                best_x_dir = x_neg.copy()
                    
                    x = best_x_dir.copy()
                    f = best_f_dir
                    
                    if f < f_min:
                        f_min = f
                        x_min = x.copy()
                        dir_min = i
                
                # After all line searches, check if we should add a new conjugate direction
                # The new direction is x_min - x_old
                x_new_dir = x_min - x_old
                
                # Check if the new direction is significantly different from existing ones
                # and if moving along it would help
                # For simplicity, we'll replace the direction that gave the worst improvement
                # with the new conjugate direction if it's useful
                
                # Compute the improvement along the new direction
                # If f_min < f_old - 1e-8, the new direction is promising
                if f_min < f_old - 1e-10:
                    # Replace the direction at dir_min with the normalized new direction
                    norm_new_dir = np.linalg.norm(x_new_dir)
                    if norm_new_dir > 1e-10:
                        directions[dir_min] = x_new_dir / norm_new_dir
                
                x = x_min.copy()
                f = f_min
                
                if f < f_best:
                    f_best = f
                    x_best = x.copy()
                
                # Convergence check
                if np.linalg.norm(x - x_old) < 1e-8:
                    break
            
            return f_best, x_best

        # Run Powell's method
        remaining = self.budget - calls_made
        powell_method(x_best, f_best, remaining)
        
        # If we have budget left, do a second pass with finer steps
        if calls_made < self.budget - 10 and time.time() < deadline:
            # Restart Powell's with a smaller initial step
            # We can modify the directions to be smaller
            # For simplicity, just run another short Powell's
            remaining2 = self.budget - calls_made
            if remaining2 > 20:
                powell_method(x_best, f_best, remaining2)
        
        return x_best, f_best
