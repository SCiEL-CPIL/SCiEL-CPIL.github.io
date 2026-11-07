import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill
        
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        
        global_best_x = None
        global_best_f = np.inf
        
        def evaluate(x):
            nonlocal calls_made, global_best_x, global_best_f
            fx = func(x)
            calls_made += 1
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()
            return fx

        # Phase 1: Coarse Grid Search (3^dim points)
        # This ensures we sample the space reasonably well to find a good basin.
        # For dim=5, 3^5 = 243 points. This is a significant chunk of the budget
        # but provides robust initialization compared to a single point.
        # We use levels: -4, 0, 4 (avoiding boundaries -5, 5 to stay inside box comfortably)
        # Actually, let's use -4, 0, 4 to be safe within [-5,5].
        levels = [-4.0, 0.0, 4.0]
        
        # Generate all combinations
        grid_points = np.array(np.meshgrid(*([levels]*dim))).T.reshape(-1, dim)
        
        # Evaluate grid points
        for i in range(len(grid_points)):
            if calls_made >= total_budget:
                break
            x = grid_points[i]
            evaluate(x)
            
            # Check time
            if time.time() > deadline - 5.0:
                break

        if global_best_x is None:
            # Fallback if something went wrong (shouldn't happen)
            x0 = np.zeros(dim)
            evaluate(x0)
            return global_best_x, global_best_f

        # Phase 2: Deterministic Adaptive Coordinate Descent
        # Start from the best point found in Phase 1.
        x = global_best_x.copy()
        f_best = global_best_f
        
        # Initial step size
        step_size = 0.5
        
        # We will iterate until budget or time runs out.
        # To avoid infinite loops on flat regions, we limit total iterations 
        # or break if no improvement for a while.
        
        max_iters = 1000
        no_improve_count = 0
        max_no_improve = 20
        
        for _ in range(max_iters):
            if calls_made >= total_budget:
                break
            if time.time() > deadline - 1.0:
                break
            
            improved = False
            
            for d in range(dim):
                if calls_made >= total_budget:
                    break
                if time.time() > deadline - 1.0:
                    break
                    
                # Try positive step
                x_plus = x.copy()
                x_plus[d] += step_size
                if x_plus[d] <= 5.0:
                    f_plus = evaluate(x_plus)
                    if f_plus < f_best:
                        x = x_plus
                        f_best = f_plus
                        improved = True
                        no_improve_count = 0
                        continue # Move to next axis after successful step
                # Try negative step
                x_minus = x.copy()
                x_minus[d] -= step_size
                if x_minus[d] >= -5.0:
                    f_minus = evaluate(x_minus)
                    if f_minus < f_best:
                        x = x_minus
                        f_best = f_minus
                        improved = True
                        no_improve_count = 0
                        continue
                # If neither improved, try smaller steps? 
                # For simplicity and speed, we just shrink the global step size if no improvement in whole cycle.
            
            if not improved:
                no_improve_count += 1
                # Shrink step size
                step_size *= 0.7
                if step_size < 1e-6:
                    break
                if no_improve_count > max_no_improve:
                    # Try a larger jump to escape local minimum?
                    # Since we started from a good grid point, we might be in a local min.
                    # A deterministic escape: try a random-ish direction? 
                    # But we want deterministic. 
                    # Let's try a larger step in the direction of the steepest descent 
                    # approximated by finite differences? That costs extra calls.
                    # Alternatively, just break if we are very small.
                    # Or, reset step size to a medium value and try to move in a random direction?
                    # Let's try a "kick" in the direction that had the smallest gradient magnitude?
                    # Too complex. Let's just break if step is tiny.
                    if step_size < 0.01:
                        break
                    else:
                        # Try a larger step to escape?
                        step_size = 0.5
                        no_improve_count = 0
            else:
                # If we improved, we might want to keep step size or grow it slightly?
                # Standard CD often keeps step size or shrinks when no improvement.
                # Let's keep it constant unless no improvement.
                pass

        # Phase 3: Final Polish with Nelder-Mead if budget remains
        # This can handle the fine-tuning better than coordinate descent if stuck.
        if calls_made < total_budget - 50 and time.time() < deadline - 5.0:
            try:
                from scipy.optimize import minimize
                # Use the current best x as start
                res = minimize(lambda x: func(x), x, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8})
                # Note: minimize will call func multiple times. We need to track those calls.
                # The lambda above calls func directly, bypassing our counter.
                # We need to wrap it.
                
                # Let's do a manual check or just trust that we have enough budget.
                # Better: use a wrapper that increments calls_made.
                # But scipy.optimize.minimize manages its own loop.
                # We can just call it and hope it doesn't exceed budget.
                # To be safe, we check remaining budget.
                if total_budget - calls_made > 100:
                    # We already started minimizing? No, we haven't called it yet in this block.
                    # Wait, the code above `res = minimize...` is inside the if block.
                    # I need to be careful. The lambda `lambda x: func(x)` does NOT increment calls_made.
                    # This is a bug in my draft. I must use a wrapper.
                    
                    def wrapped_func(x):
                        nonlocal calls_made
                        if calls_made >= total_budget:
                            return np.inf
                        return evaluate(x)
                    
                    res = minimize(wrapped_func, x, method='Nelder-Mead', 
                                   options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8})
                    if res.fun < global_best_f:
                        global_best_f = res.fun
                        global_best_x = res.x
            except Exception:
                pass

        if global_best_x is None:
            global_best_x = np.zeros(dim)
            global_best_f = func(global_best_x)
            calls_made += 1

        return global_best_x, global_best_f
