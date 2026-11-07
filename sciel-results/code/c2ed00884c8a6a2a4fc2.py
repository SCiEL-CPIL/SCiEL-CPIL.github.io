import numpy as np
import time

class Optimizer:
    """
    Powell's Method Optimizer for BBOB-style functions.
    Handles anisotropy by rotating the coordinate system to align with the function's principal axes.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 290.0  # Safety margin before 300s kill
        
        # Initialization
        x = np.zeros(self.dim)
        fx = func(x)
        
        # Directions: Standard basis vectors
        directions = [np.eye(self.dim)[i] for i in range(self.dim)]
        
        # Count evaluations
        eval_count = 1
        max_evals = self.budget
        
        # Helper: Safe evaluation with budget and time check
        def safe_eval(x_new):
            nonlocal eval_count
            if eval_count >= max_evals:
                return None
            if time.time() - start_time > time_limit:
                return None
            try:
                val = func(x_new)
                eval_count += 1
                return val
            except:
                return None

        # Helper: 1D Golden Section Search
        def golden_section_search(func_1d, a, b, max_iter=30):
            """
            Minimizes func_1d(x) in [a, b].
            Returns best x and f(x).
            """
            phi = (np.sqrt(5) - 1) / 2
            c = b - phi * (b - a)
            d = a + phi * (b - a)
            fc = func_1d(c)
            fd = func_1d(d)
            
            if fc is None or fd is None:
                # If we hit limits, return the best we have
                if fc is not None and (fd is None or fc < fd):
                    return c, fc
                elif fd is not None:
                    return d, fd
                mid = (a+b)/2
                fm = func_1d(mid)
                if fm is not None:
                    return mid, fm
                return (a+b)/2, None

            for _ in range(max_iter):
                if time.time() - start_time > time_limit:
                    break
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - phi * (b - a)
                    fc = func_1d(c)
                    if fc is None:
                        break
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + phi * (b - a)
                    fd = func_1d(d)
                    if fd is None:
                        break
            
            # Return the best of the last points
            if fc is not None and fd is not None:
                if fc < fd:
                    return c, fc
                else:
                    return d, fd
            elif fc is not None:
                return c, fc
            elif fd is not None:
                return d, fd
            else:
                return (a+b)/2, None

        # Helper: Perform 1D minimization along direction d from point x0
        def minimize_1d(x0, d, f0):
            """
            Finds best x = x0 + alpha*d.
            Returns x_best, f_best, alpha_best.
            """
            # Define 1D function
            def f_1d(alpha):
                x_new = x0 + alpha * d
                return safe_eval(x_new)
            
            # We are at a local minimum along d from previous iteration?
            # Not necessarily for d_new.
            # For d_i, yes.
            # Strategy: Expand from 0.
            
            # Check current
            # f0 is known, but let's re-evaluate to be safe? No, save budget.
            # f(0) = f0.
            
            # Try small step
            step = 1.0
            a, b = 0.0, 0.0
            f_a = f0
            
            # Determine direction of descent
            f_pos = f_1d(step)
            f_neg = f_1d(-step)
            
            if f_pos is None and f_neg is None:
                return x0, f0, 0.0
            
            # If one is None, use the other if it's better
            if f_pos is not None and (f_neg is None or f_pos < f_neg):
                if f_pos < f0:
                    # Descending positive
                    alpha = step
                    f_curr = f_pos
                    # Expand
                    while True:
                        alpha_next = 2.0 * alpha
                        f_next = f_1d(alpha_next)
                        if f_next is None or f_next >= f_curr:
                            break
                        alpha = alpha_next
                        f_curr = f_next
                    a, b = 0.0, alpha
                else:
                    return x0, f0, 0.0
            elif f_neg is not None and (f_pos is None or f_neg < f_pos):
                if f_neg < f0:
                    # Descending negative
                    alpha = -step
                    f_curr = f_neg
                    # Expand
                    while True:
                        alpha_next = 2.0 * alpha
                        f_next = f_1d(alpha_next)
                        if f_next is None or f_next >= f_curr:
                            break
                        alpha = alpha_next
                        f_curr = f_next
                    a, b = alpha, 0.0
                else:
                    return x0, f0, 0.0
            else:
                # Both worse or equal
                return x0, f0, 0.0

            if a > b:
                a, b = b, a
            
            # Golden Section
            alpha_opt, f_opt = golden_section_search(f_1d, a, b, max_iter=20)
            
            if f_opt is not None and f_opt < f0:
                x_opt = x0 + alpha_opt * d
                return x_opt, f_opt, alpha_opt
            else:
                return x0, f0, 0.0

        # Main Loop
        while eval_count < max_evals and (time.time() - start_time < time_limit):
            x_start = x.copy()
            f_start = fx
            
            improvements = []
            worst_idx = 0
            worst_improvement = np.inf
            
            # 1. Minimize along each existing direction
            for i, d in enumerate(directions):
                x_new, f_new, _ = minimize_1d(x, d, fx)
                x = x_new
                fx = f_new
                improvement = f_start - fx # Wait, improvement should be relative to start of this direction?
                # Standard Powell's: improvement is f(before) - f(after) for this direction.
                # But I need to track the "worst" improvement in the cycle.
                # Let's track improvement relative to the state before this direction's search.
                
                # Correction:
                # I need to store f_before_dir.
                
            # Re-doing the loop with correct tracking
            x_cycle_start = x_start
            f_cycle_start = f_start
            
            improvements = []
            worst_idx = 0
            worst_improvement = np.inf
            
            for i, d in enumerate(directions):
                f_before_dir = fx
                x_new, f_new, _ = minimize_1d(x, d, fx)
                x = x_new
                fx = f_new
                improvement = f_before_dir - fx
                improvements.append(improvement)
                
                if improvement < worst_improvement:
                    worst_improvement = improvement
                    worst_idx = i
            
            if eval_count >= max_evals or time.time() - start_time > time_limit:
                break
            
            # 2. Determine new direction
            d_new = x - x_cycle_start
            norm_d_new = np.linalg.norm(d_new)
            
            if norm_d_new > 1e-10:
                # 3. Minimize along new direction from x_cycle_start
                x_final, f_final, _ = minimize_1d(x_cycle_start, d_new, f_cycle_start)
                
                # 4. Check if new direction is better than worst
                improvement_new = f_cycle_start - f_final
                
                if improvement_new > worst_improvement:
                    # Replace worst direction
                    d_normalized = d_new / norm_d_new
                    directions[worst_idx] = d_normalized
                    
                # Update current best to the result of the new direction search
                # Note: x_final is likely better than x (which was the end of the cycle)
                # because we optimized along the direct path.
                x = x_final
                fx = f_final

        return x, fx
