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
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        def nelder_mead(func, x0, initial_step, max_evals, time_check, eval_count_ref):
            """
            Lightweight Nelder-Mead implementation.
            Returns (best_x, best_f, evals_used)
            """
            n = dim
            # Initialize simplex
            simplex = [x0.copy() for _ in range(n + 1)]
            # First vertex is x0, others are x0 + step * e_i
            for i in range(n):
                simplex[i+1][i] += initial_step
            
            # Evaluate initial simplex
            values = []
            for i in range(n + 1):
                if eval_count_ref[0] >= budget or time_check():
                    break
                f_val = func(simplex[i])
                eval_count_ref[0] += 1
                values.append(f_val)
                if f_val < best_f:
                    best_f = f_val
                    best_x = simplex[i].copy()
            
            if len(values) < n + 1:
                return best_x, best_f, 0

            # Standard NM parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            iterations = 0
            max_iterations = max_evals # Rough cap on iterations, not evals
            
            while eval_count_ref[0] < budget and not time_check() and iterations < max_iterations:
                iterations += 1
                
                # Sort simplex
                order = np.argsort(values)
                simplex = [simplex[i] for i in order]
                values = [values[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                if eval_count_ref[0] >= budget or time_check():
                    break
                f_r = func(x_r)
                eval_count_ref[0] += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                    
                if f_r < values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    if eval_count_ref[0] >= budget or time_check():
                        break
                    f_e = func(x_e)
                    eval_count_ref[0] += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                        
                    if f_e < f_r:
                        simplex[-1] = x_e
                        values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        values[-1] = f_r
                elif f_r < values[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    values[-1] = f_r
                else:
                    # Contraction
                    if f_r < values[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                        
                    if eval_count_ref[0] >= budget or time_check():
                        break
                    f_c = func(x_c)
                    eval_count_ref[0] += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                        
                    if f_c < min(f_r, values[-1]):
                        simplex[-1] = x_c
                        values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if eval_count_ref[0] >= budget or time_check():
                                break
                            f_s = func(simplex[i])
                            eval_count_ref[0] += 1
                            values[i] = f_s
                            if f_s < best_f:
                                best_f = f_s
                                best_x = simplex[i].copy()
                        
                        if eval_count_ref[0] >= budget or time_check():
                            break
                        
                        # Check for convergence
                        spread = np.max(values) - np.min(values)
                        if spread < 1e-8:
                            break

            return best_x, best_f, iterations

        # Multi-start strategy
        # We have 500 budget.
        # NM requires n+1 = 6 evals to start.
        # Let's try 10 starts. 10 * 6 = 60 evals. Remaining 440 for optimization.
        # Or fewer starts with more budget each?
        # Step function has flat regions. NM might get stuck on a plateau if step size is wrong.
        # Adaptive step size: start large, shrink if no improvement.
        
        num_starts = 10
        # Distribute budget among starts
        # First start gets more budget? Or equal?
        # Let's do equal split of remaining budget after initial simplex setup.
        # Actually, just run NM until time/budget runs out, then restart if time remains.
        
        # Initial points: Origin and random points
        start_points = [np.zeros(dim)]
        for _ in range(num_starts - 1):
            start_points.append(rng.uniform(-4.5, 4.5, dim))
            
        initial_step = 1.5
        
        for i, x0 in enumerate(start_points):
            if eval_count >= budget or time_check():
                break
            
            # Allocate budget for this start
            # If it's the last start, give it all remaining.
            # Otherwise, give it a portion.
            remaining_starts = num_starts - i
            if remaining_starts == 1:
                max_evals_for_start = budget - eval_count
            else:
                # Heuristic: give more to earlier starts? Or just split?
                # Let's split remaining budget equally among remaining starts, minus some buffer for simplex init
                # Actually, NM is fast. Let's just let it run until a local min is found or budget runs low.
                # But we need to ensure we try other starts.
                # Let's reserve budget.
                # 500 / 10 = 50 evals per start.
                # 6 for init + 44 for optimization.
                # This is quite tight.
                # Maybe fewer starts? 5 starts? 500/5 = 100.
                # Let's try 5 starts.
                pass 

        # Re-evaluating strategy: 5 starts is safer for budget.
        num_starts = 5
        start_points = [np.zeros(dim)]
        for _ in range(num_starts - 1):
            start_points.append(rng.uniform(-4.5, 4.5, dim))
            
        initial_step = 2.0
        
        for i, x0 in enumerate(start_points):
            if eval_count >= budget or time_check():
                break
            
            # Reset best for this local search? No, global best is tracked.
            # But NM uses local values.
            # We pass a mutable eval_count list.
            eval_count_ref = [eval_count]
            
            # Max evals for this start
            # If last start, use all remaining.
            # Else, use (budget - eval_count) / (num_starts - i)
            remaining_starts = num_starts - i
            if remaining_starts == 1:
                max_evals = budget - eval_count
            else:
                # Keep some buffer
                max_evals = int((budget - eval_count) / remaining_starts)
                # Ensure at least enough for simplex + few steps
                if max_evals < 20:
                    max_evals = budget - eval_count
            
            # Run NM
            # We don't really use the returned best_x/f from NM directly for global best update
            # because we update global best inside NM loop via closure.
            # Wait, the closure in NM updates the OUTER best_x/best_f.
            # This is correct.
            
            nelder_mead(func, x0, initial_step, max_evals, time_check, eval_count_ref)
            
            # Update eval_count from ref
            eval_count = eval_count_ref[0]
            
            # If we improved significantly, maybe reduce step size for next start?
            # Or if we didn't improve, increase step size?
            # For now, fixed step size.
            
            # Check if we are stuck
            if time_check():
                break

        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            if eval_count < budget:
                best_f = func(best_x)
                eval_count += 1
            else:
                best_f = np.inf

        return best_x, best_f
