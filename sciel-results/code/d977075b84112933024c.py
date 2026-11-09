"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start Powell Method optimizer for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        dim = self.dim
        total_budget = self.budget
        
        global_best_x = None
        global_best_f = float('inf')
        calls_used = 0
        
        # Helper to update global best
        def update_best(x, f):
            nonlocal global_best_x, global_best_f, calls_used
            if f < global_best_f:
                global_best_f = f
                global_best_x = x.copy()
            # Note: calls_used is incremented externally by the caller after func returns
        
        # Helper to check time
        def time_up():
            return time.time() > deadline - 5.0

        # Helper to check budget
        def budget_up():
            return calls_used >= total_budget

        # Powell's Method Implementation
        # Powell's method minimizes f(x) by successively minimizing along a set of directions.
        # It starts with orthogonal directions (usually the coordinate axes) and updates them
        # to be more aligned with the valleys of the function.
        
        def powell_method(x0, max_iter=50):
            nonlocal calls_used, global_best_x, global_best_f
            
            # Initial directions: standard basis vectors
            # We use unit vectors to start
            directions = np.eye(dim)
            x = x0.copy()
            
            # Evaluate current point
            if budget_up() or time_up():
                return x, global_best_f
            
            fx = func(x)
            calls_used += 1
            update_best(x, fx)
            
            for _ in range(max_iter):
                if budget_up() or time_up():
                    break
                    
                x_old = x.copy()
                f_old = fx
                f_min = fx
                min_dir_idx = 0
                
                # Step 1: Minimize along each direction
                for i in range(dim):
                    if budget_up() or time_up():
                        return x, global_best_f
                    
                    dir_vec = directions[i]
                    # Find minimum along the line x + t * dir_vec
                    # Use a simple ternary search or golden section search
                    # Since the domain is bounded [-5, 5], we can bound the search.
                    # However, the direction might point out of bounds.
                    # We clip the resulting x to [-5, 5].
                    
                    # Determine range for t
                    # Let's try a small range around 0, expanding if needed?
                    # Or just use a fixed range like [-1, 1] initially?
                    # Given the scale of the problem, let's use a robust line search.
                    
                    # Golden Section Search on t in [t_low, t_high]
                    # We need to estimate a good interval.
                    # Let's try t in [-1.0, 1.0] first. If the function decreases at the boundary, expand.
                    t_low = -1.0
                    t_high = 1.0
                    
                    # Check if we are at the boundary of the box
                    # If x[i] is close to 5, we can only go negative. If -5, only positive.
                    # This is handled by clipping x.
                    
                    # To save evaluations, let's do a simple quadratic fit or just a few points?
                    # Given the budget constraints (500 calls), we should be efficient.
                    # Let's use a fixed number of evaluations for line search, e.g., 5-8.
                    # Or use scipy.optimize.minimize_scalar? No, let's keep it simple and robust.
                    
                    # Let's use a simple grid search for the line search step size to keep it fast and deterministic.
                    # Or better, since we have a limited budget, let's use a fixed step size approach 
                    # combined with the Powell update logic.
                    
                    # Actually, standard Powell's requires finding the minimum along the line.
                    # Let's use a small number of evaluations for the line search.
                    # Let's try 4 points in the interval [-1, 1] and pick the best, then refine?
                    # Or just use the best of a few candidates.
                    
                    # Candidate t values: 0, 0.5, -0.5, 1.0, -1.0
                    # This is 5 evaluations per direction per iteration.
                    # 5 directions * 5 evals * 50 iterations = 1250 evals. Too many.
                    
                    # We need to be much more efficient.
                    # Let's limit the line search to 3 evaluations per direction per iteration.
                    # t = 0.5, -0.5, 0.0 (current)
                    # Actually, we already have f(x) for t=0.
                    # Let's evaluate at t=0.5 and t=-0.5.
                    # Pick the best.
                    
                    x_pos = x + 0.5 * dir_vec
                    x_pos = np.clip(x_pos, -5, 5)
                    x_neg = x - 0.5 * dir_vec
                    x_neg = np.clip(x_neg, -5, 5)
                    
                    f_pos = func(x_pos)
                    calls_used += 1
                    update_best(x_pos, f_pos)
                    
                    f_neg = func(x_neg)
                    calls_used += 1
                    update_best(x_neg, f_neg)
                    
                    # Select the best of current, pos, neg
                    if f_pos < f_neg and f_pos < fx:
                        x_new = x_pos
                        f_new = f_pos
                    elif f_neg < f_pos and f_neg < fx:
                        x_new = x_neg
                        f_new = f_neg
                    else:
                        x_new = x
                        f_new = fx
                        
                    if f_new < f_min:
                        f_min = f_new
                        min_dir_idx = i
                        
                    x = x_new
                    fx = f_new
                
                # Step 2: Check for convergence
                if abs(f_old - f_min) < 1e-8:
                    break
                    
                # Step 3: Update directions
                # The direction that gave the largest decrease is replaced by the direction 
                # from the start of the iteration to the end of the iteration.
                new_dir = x - x_old
                norm_new_dir = np.linalg.norm(new_dir)
                if norm_new_dir > 1e-10:
                    new_dir /= norm_new_dir
                else:
                    new_dir = directions[min_dir_idx]
                
                directions[min_dir_idx] = new_dir
                
                # Optional: If the decrease in the last step was small, stop?
                # The convergence check above handles this.
                
            return x, fx

        # Phase 1: Multi-Start
        # Generate a set of random starting points.
        # We have 500 calls.
        # Each Powell run uses ~2 calls per direction per iteration.
        # Let's say we run Powell for 10 iterations.
        # 10 iters * 5 dirs * 2 calls/iter = 100 calls per run.
        # We can afford ~4-5 runs.
        
        # Let's use 5 random starts.
        # Also include the center (0,0,0,0,0) as one of the starts.
        
        rng = np.random.default_rng(42)
        starts = []
        starts.append(np.zeros(dim))
        for _ in range(4):
            starts.append(rng.uniform(-5, 5, dim))
        
        # Shuffle starts to avoid bias
        perm = rng.permutation(len(starts))
        starts = [starts[i] for i in perm]
        
        for x0 in starts:
            if budget_up() or time_up():
                break
            
            # Run Powell's method
            # Limit iterations based on remaining budget
            # Est. 10 calls per iteration (5 dirs * 2 calls)
            # Remaining calls / 10
            remaining = total_budget - calls_used
            max_iter = max(5, remaining // 10)
            
            _, _ = powell_method(x0, max_iter=max_iter)
            
            # If we still have significant budget, we could do another round from the best point found so far?
            # Or just let the loop handle it.
            
        # If we have budget left, do a final refinement on the global best
        if not budget_up() and not time_up() and global_best_x is not None:
            remaining = total_budget - calls_used
            max_iter = max(5, remaining // 10)
            # Run one more Powell from the global best to polish
            _, _ = powell_method(global_best_x, max_iter=max_iter)

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
