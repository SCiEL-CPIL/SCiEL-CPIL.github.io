import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        n_calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        
        # To handle the "clipping" in the evaluator, we ensure our internal points stay in bounds
        # The evaluator clips, so we can just evaluate any point, but staying in bounds
        # reduces wasted evaluations on the boundary if the min is interior.
        
        def call_f(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None
            n_calls += 1
            # x is assumed to be a numpy array
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def powell_method(x0, max_calls, tol=1e-10):
            """
            Powell's method for minimizing a function in low dimensions.
            """
            nonlocal n_calls, best_x, best_f
            
            if n_calls >= max_calls:
                return best_x, best_f, n_calls

            # Initialize directions as standard basis
            directions = np.eye(dim)
            x = x0.copy()
            f_x = call_f(x)
            if f_x is None:
                return best_x, best_f, n_calls

            # Line search along a direction
            def line_search(x, d, f_x, max_calls_local):
                """
                Parabolic interpolation line search.
                """
                nonlocal n_calls
                if n_calls >= max_calls_local:
                    return x, f_x, 0
                
                # Initial step size
                step = 0.5
                # Try to find a better point
                x_trial = x + step * d
                x_trial = np.clip(x_trial, -5.0, 5.0)
                f_trial = call_f(x_trial)
                if f_trial is None:
                    return x, f_x, 0
                
                calls_used = 1
                
                if f_trial < f_x:
                    # Move in this direction
                    x = x_trial
                    f_x = f_trial
                    # Try to expand
                    while n_calls < max_calls_local:
                        x_next = x + step * d
                        x_next = np.clip(x_next, -5.0, 5.0)
                        f_next = call_f(x_next)
                        if f_next is None:
                            break
                        calls_used += 1
                        if f_next < f_x:
                            x = x_next
                            f_x = f_next
                            step *= 1.5
                        else:
                            # Parabolic interpolation
                            # We have points: x - step*d (f_prev, not stored explicitly, 
                            # but we can approximate or just use bisection if we stored it)
                            # For simplicity and robustness, let's use a simple bisection
                            # if we detect a peak, or just break.
                            # Actually, standard Powell uses parabolic fit.
                            # Let's stick to a simpler robust approach: 
                            # If next is worse, try half step.
                            x_half = x + 0.5 * step * d
                            x_half = np.clip(x_half, -5.0, 5.0)
                            f_half = call_f(x_half)
                            if f_half is None:
                                break
                            calls_used += 1
                            if f_half < f_x:
                                x = x_half
                                f_x = f_half
                                step *= 0.5
                            else:
                                break
                else:
                    # Try negative direction
                    x_trial_neg = x - step * d
                    x_trial_neg = np.clip(x_trial_neg, -5.0, 5.0)
                    f_trial_neg = call_f(x_trial_neg)
                    if f_trial_neg is None:
                        return x, f_x, calls_used + 1
                    calls_used += 1
                    if f_trial_neg < f_x:
                        x = x_trial_neg
                        f_x = f_trial_neg
                        step *= 1.5
                        while n_calls < max_calls_local:
                            x_next = x - step * d
                            x_next = np.clip(x_next, -5.0, 5.0)
                            f_next = call_f(x_next)
                            if f_next is None:
                                break
                            calls_used += 1
                            if f_next < f_x:
                                x = x_next
                                f_x = f_next
                                step *= 1.5
                            else:
                                x_half = x - 0.5 * step * d
                                x_half = np.clip(x_half, -5.0, 5.0)
                                f_half = call_f(x_half)
                                if f_half is None:
                                    break
                                calls_used += 1
                                if f_half < f_x:
                                    x = x_half
                                    f_x = f_half
                                    step *= 0.5
                                else:
                                    break
                    else:
                        # No improvement in either direction, reduce step size and try again?
                        # Or just accept this direction yielded 0 improvement.
                        step *= 0.5
                        if step < 1e-6:
                            break
                        # Try one more time with smaller step
                        x_trial_small = x + step * d
                        x_trial_small = np.clip(x_trial_small, -5.0, 5.0)
                        f_trial_small = call_f(x_trial_small)
                        if f_trial_small is None:
                            break
                        calls_used += 1
                        if f_trial_small < f_x:
                             x = x_trial_small
                             f_x = f_trial_small
                        else:
                             break

                return x, f_x, calls_used

            # Main Powell Loop
            for _ in range(20): # Max cycles
                if n_calls >= max_calls or time.time() > deadline - 5:
                    break
                
                x_start_cycle = x.copy()
                f_start_cycle = f_x
                max_improvement = 0.0
                max_improvement_dir = 0
                
                for i in range(dim):
                    if n_calls >= max_calls or time.time() > deadline - 5:
                        break
                    
                    x, f_x, _ = line_search(x, directions[i], f_x, max_calls)
                    improvement = f_start_cycle - f_x # Wait, f_x changes in loop. 
                    # We need improvement relative to start of this specific line search?
                    # Standard Powell: improvement is f_start_of_line - f_end_of_line
                    # But we lose that. Let's just track global improvement in cycle.
                    
                    # Actually, let's track the direction with the largest drop in f
                    # We need to store f before line search.
                    # Let's refactor line_search to return the drop.
                    # For now, this implementation is a bit loose. 
                    # Let's use a simpler, more robust Powell implementation.

            # The above line_search is getting messy. Let's use a standard robust Powell implementation.
            pass

        # Given the complexity of writing a perfect Powell's method in a short snippet,
        # and the success of Nelder-Mead in Inspiration 1, I will stick to a highly tuned 
        # Multi-Start Nelder-Mead with a final CMA-ES-like polish.
        
        def nelder_mead(x0, max_calls, step_size=0.5):
            nonlocal n_calls, best_x, best_f
            
            if n_calls >= max_calls:
                return best_x, best_f, n_calls

            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(1, dim + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += step_size
            
            fvals = np.zeros(dim + 1)
            for i in range(dim + 1):
                f = call_f(simplex[i])
                if f is None:
                    return best_x, best_f, n_calls
                fvals[i] = f
            
            def sort_simplex():
                nonlocal simplex, fvals
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]
            
            sort_simplex()
            
            # Main loop
            while n_calls < max_calls and time.time() < deadline - 5:
                if np.std(fvals) < 1e-10:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + 1.0 * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = call_f(x_r)
                if f_r is None:
                    break
                
                if fvals[0] <= f_r < fvals[-1]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                    sort_simplex()
                elif f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + 2.0 * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = call_f(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                    sort_simplex()
                else:
                    # Contraction
                    x_c = centroid + 0.5 * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = call_f(x_c)
                    if f_c is None:
                        break
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                        sort_simplex()
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            f = call_f(simplex[i])
                            if f is None:
                                return best_x, best_f, n_calls
                            fvals[i] = f
                        sort_simplex()
                        
            return best_x, best_f, n_calls

        # Strategy:
        # 1. Run NM from 0 with budget 200.
        # 2. Run NM from best_x with smaller step size and budget 100.
        # 3. Run NM from random points if budget remains.
        # 4. Final (1+1)-ES polish.

        # Run 1: Global
        x_start = np.zeros(dim)
        nelder_mead(x_start, 200, step_size=1.0)
        
        # Run 2: Local refinement on best found
        x_start = best_x.copy()
        nelder_mead(x_start, 100, step_size=0.1)
        
        # Run 3-5: Random restarts if budget allows
        remaining_budget = budget - n_calls
        if remaining_budget > 50:
             x_start = rng.uniform(-3, 3, dim)
             nelder_mead(x_start, min(100, remaining_budget - 10), step_size=0.5)
             
        # Final Polish with (1+1)-ES
        remaining = budget - n_calls
        if remaining > 5 and time.time() < deadline - 2:
            x = best_x.copy()
            fx = best_f
            sigma_poly = 0.05
            for _ in range(remaining):
                if time.time() > deadline - 1:
                    break
                y = np.clip(x + sigma_poly * rng.standard_normal(dim), -5.0, 5.0)
                fy = call_f(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma_poly *= 1.05
                    if fx < best_f:
                        best_x = x.copy()
                        best_f = fx
                else:
                    sigma_poly *= 0.9
                    sigma_poly = max(sigma_poly, 1e-5)

        return best_x, best_f
