import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Time guard: evaluator kills at 330s, leave 40s safety margin
        start_time = time.time()
        deadline = start_time + 290

        best_x = np.zeros(dim)
        best_f = np.inf
        evals_used = 0
        eval_log = []  # (x, f) for all evaluations

        def try_func(x):
            nonlocal evals_used, best_f, best_x
            if evals_used >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(x)
            if not np.isfinite(f_val):
                return np.inf
            evals_used += 1
            eval_log.append((x.copy(), f_val))
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def nelder_mead_adaptive(x_start, max_iter=200, initial_step=1.0):
            """Adaptive Nelder-Mead that adjusts step size based on function behavior."""
            nonlocal evals_used, best_f, best_x, deadline
            
            if time.time() > deadline or evals_used >= budget:
                return best_x, best_f

            # Initialize simplex with adaptive initial step
            step = initial_step
            simplex = [x_start.copy()]
            for i in range(dim):
                xi = x_start.copy()
                xi[i] += step
                simplex.append(xi)
            
            # Evaluate initial simplex
            fs = []
            for i in range(dim + 1):
                if evals_used >= budget or time.time() > deadline:
                    return best_x, best_f
                f_val = try_func(simplex[i])
                fs.append(f_val)
            
            # Nelder-Mead parameters
            alpha_nm = 1.0  # reflection
            gamma = 2.0     # expansion
            rho = 0.5       # contraction
            sigma = 0.5     # shrink
            
            # Track best step size for adaptation
            best_step = step
            stagnant = 0
            
            for it in range(max_iter):
                if evals_used >= budget or time.time() > deadline:
                    break
                
                # Sort simplex by function value
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]
                
                # Check for convergence: if all values are very close
                f_range = fs[-1] - fs[0]
                if f_range < 1e-10:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:dim], axis=0)
                
                # Reflection
                x_ref = centroid + alpha_nm * (centroid - simplex[-1])
                f_ref = try_func(x_ref)
                
                improved = False
                if f_ref < fs[0]:
                    # Try expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    f_exp = try_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fs[-1] = f_exp
                        improved = True
                        # If expansion works well, increase step
                        if f_exp < fs[0] * 0.9:
                            best_step = min(best_step * 1.5, 5.0)
                    else:
                        simplex[-1] = x_ref
                        fs[-1] = f_ref
                        improved = True
                elif f_ref < fs[-2]:
                    simplex[-1] = x_ref
                    fs[-1] = f_ref
                    improved = True
                else:
                    # Contraction
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    f_con = try_func(x_con)
                    if f_con < fs[-1]:
                        simplex[-1] = x_con
                        fs[-1] = f_con
                        improved = True
                    else:
                        # Shrink all points toward best
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fs[i] = try_func(simplex[i])
                        if evals_used >= budget or time.time() > deadline:
                            break
                        # If shrinking, decrease step
                        best_step = max(best_step * 0.5, 0.01)
                
                if not improved:
                    stagnant += 1
                    if stagnant > 5:
                        # Reset step size and try again
                        step = max(best_step * 0.5, 0.01)
                        stagnant = 0
                else:
                    stagnant = 0
                
                # Update global best
                for i in range(dim + 1):
                    if fs[i] < best_f:
                        best_f = fs[i]
                        best_x = simplex[i].copy()
                
                # Adaptive step: if we're making progress, keep current; if not, shrink
                if it % 10 == 0 and it > 0:
                    # Check if simplex is still large
                    simplex_size = np.max(np.abs(np.array(simplex) - np.array(simplex[0])), axis=0)
                    if np.max(simplex_size) > 0.1:
                        # Still exploring, keep step
                        pass
                    else:
                        # Converging, shrink step
                        best_step = max(best_step * 0.8, 0.001)
            
            return best_x, best_f

        # Phase 1: Coarse exploration to find the cliff region
        # The step-ellipsoidal function is flat until close to the minimum
        # We need to find where the function value drops sharply
        
        # Generate a coarse grid in [-4, 4]^5 (minimizer is in this range)
        # Use 3 points per dimension: -4, 0, 4
        # That's 3^5 = 243 points, too many. Use a subset.
        
        # Strategy: Start with center, then expand outward
        # First evaluate center
        try_func(np.zeros(dim))
        
        # Then evaluate points along each axis at distance 2 and 4
        axis_points = []
        for i in range(dim):
            for dist in [2.0, 4.0]:
                for sign in [1, -1]:
                    x = np.zeros(dim)
                    x[i] = sign * dist
                    axis_points.append(x)
        
        # Evaluate axis points
        for x in axis_points:
            if evals_used >= budget or time.time() > deadline:
                break
            try_func(x)
        
        # Now we have a sense of where the function changes
        # Find the best point from phase 1
        if evals_used >= budget or time.time() > deadline:
            return best_x, best_f
        
        # Phase 2: Local refinement around the best region
        # Use adaptive Nelder-Mead from the best point found
        # Also try from a few other promising points
        
        # Sort eval_log by function value to find top candidates
        eval_log_sorted = sorted(eval_log, key=lambda t: t[1])
        
        # Get top 5 distinct points (by location, not just value)
        top_points = []
        for x, f in eval_log_sorted:
            # Check if this point is sufficiently different from existing top points
            is_new = True
            for tp in top_points:
                if np.linalg.norm(x - tp) < 0.5:
                    is_new = False
                    break
            if is_new:
                top_points.append(x)
            if len(top_points) >= 5:
                break
        
        # Run adaptive Nelder-Mead from each top point
        for i, x_start in enumerate(top_points):
            if evals_used >= budget or time.time() > deadline:
                break
            
            # Use larger initial step for earlier runs, smaller for later
            initial_step = max(1.0 - i * 0.2, 0.3)
            max_iter = max(20, (budget - evals_used) // (dim + 1) // (5 - i))
            
            nelder_mead_adaptive(x_start.copy(), max_iter=max_iter, initial_step=initial_step)
        
        # Phase 3: If budget remains, do final refinement from best
        if evals_used < budget and time.time() < deadline:
            remaining = budget - evals_used
            if remaining > 10:
                # Final polish with very small steps
                nelder_mead_adaptive(best_x.copy(), max_iter=min(100, remaining // (dim + 1)), initial_step=0.1)
        
        return best_x, best_f
