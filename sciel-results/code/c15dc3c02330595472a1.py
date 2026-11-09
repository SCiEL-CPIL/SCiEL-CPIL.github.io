import numpy as np
import time

class Optimizer:
    """
    Multi-start Nelder-Mead optimizer for BBOB-style f19.
    
    Strategy:
    1. Evaluate a diverse set of initial points to identify promising basins.
    2. Run Nelder-Mead local search from the best initial points.
    3. Polish the global best with a final local search if budget remains.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = 325.0  # Safety margin before 330s kill
        dim = self.dim
        total_budget = self.budget
        used = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def check_deadline():
            return time.time() > deadline

        def evaluate(x):
            nonlocal used, best_f, best_x
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # --- Phase 1: Diverse Initialization ---
        # We want to spread out the initial evaluations to quickly lower b_i.
        # Use a combination of deterministic and random points.
        rng = np.random.default_rng(42)
        
        # Deterministic points
        init_points = [np.zeros(dim)]
        # Axis-aligned points
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 4.0
                init_points.append(p)
        
        # Random points
        # Allocate 20% of budget to initialization, min 10 points
        init_budget = max(10, total_budget // 5)
        num_random = init_budget - len(init_points)
        for _ in range(max(0, num_random)):
            p = rng.uniform(-4.5, 4.5, dim)
            init_points.append(p)
            
        # Evaluate all initial points
        init_vals = []
        for p in init_points:
            if check_deadline() or used >= total_budget:
                break
            f_val = evaluate(p)
            init_vals.append(f_val)
            
        if not init_vals:
            return best_x, best_f

        # Sort initial points by fitness
        sorted_indices = np.argsort(init_vals)
        
        # --- Phase 2: Multi-Start Nelder-Mead ---
        # We will run Nelder-Mead from the top K initial points.
        # K is chosen to balance exploration and exploitation.
        # Each run gets a limited budget.
        
        # Simplex size: Start with a relatively large step to escape shallow local minima
        # but not so large that we jump out of the basin.
        # For 5D, a step size of 0.5-1.0 is reasonable.
        step_size = 0.5
        
        # Number of starts: Let's try 5 best starts.
        num_starts = min(5, len(init_vals))
        
        # Budget per start: Remaining budget / num_starts
        remaining_budget = total_budget - used
        budget_per_start = remaining_budget // num_starts if num_starts > 0 else remaining_budget
        
        # Nelder-Mead Parameters
        # alpha: reflection, beta: expansion, gamma: contraction, delta: shrink
        alpha = 1.0
        beta = 2.0
        gamma = 0.5
        delta = 0.5
        
        for start_idx in range(num_starts):
            if check_deadline() or used >= total_budget:
                break
            
            x0 = init_points[sorted_indices[start_idx]].copy()
            f0 = init_vals[sorted_indices[start_idx]]
            
            # Initialize Simplex
            # Vertices: x0, x0 + step_size * e_i
            simplex = [x0.copy()]
            for i in range(dim):
                v = x0.copy()
                v[i] += step_size
                simplex.append(v)
                
            # Evaluate initial simplex vertices (except x0 which is already evaluated)
            simplex_vals = [f0]
            for i in range(1, dim + 1):
                if check_deadline() or used >= total_budget:
                    break
                f_val = evaluate(simplex[i])
                simplex_vals.append(f_val)
                
            # If we ran out of budget during simplex init, break
            if len(simplex_vals) < dim + 1:
                break
                
            # Nelder-Mead Iteration
            # Limit iterations per start to avoid getting stuck
            max_iters = budget_per_start
            # We need at least dim+1 evaluations per iteration roughly, so max_iters is a loose bound on evaluations
            # Actually, let's count evaluations directly.
            
            local_used = 0
            while local_used < budget_per_start and not check_deadline() and used < total_budget:
                # Sort simplex by function value
                order = np.argsort(simplex_vals)
                simplex = [simplex[i] for i in order]
                simplex_vals = [simplex_vals[i] for i in order]
                
                # Check convergence: if the difference between best and worst is small
                # and the simplex size is small, we can stop this local search.
                if simplex_vals[-1] - simplex_vals[0] < 1e-6:
                    # Check simplex size
                    centroid = np.mean(simplex[:-1], axis=0)
                    max_dist = max(np.linalg.norm(s - centroid) for s in simplex)
                    if max_dist < 1e-4:
                        break
                        # Otherwise, continue to shrink if possible
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst_point = simplex[-1]
                worst_val = simplex_vals[-1]
                
                # Reflection
                x_r = centroid + alpha * (centroid - worst_point)
                f_r = evaluate(x_r)
                local_used += 1
                
                if f_r < simplex_vals[0]:
                    # Expansion
                    x_e = centroid + beta * (centroid - worst_point)
                    f_e = evaluate(x_e)
                    local_used += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_vals[-1] = f_r
                elif f_r < simplex_vals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_vals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + gamma * (worst_point - centroid)
                    f_c = evaluate(x_c)
                    local_used += 1
                    if f_c < worst_val:
                        simplex[-1] = x_c
                        simplex_vals[-1] = f_c
                    else:
                        # Shrink
                        # Move all points (except best) halfway towards the best
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + delta * (simplex[i] - simplex[0])
                            # Re-evaluate? Nelder-Mead usually re-evaluates on shrink.
                            # To save budget, we can just update values if we assume they get worse,
                            # but standard NM re-evaluates. Let's re-evaluate to be safe, 
                            # but this is expensive. 
                            # Alternative: Just break if shrinking is needed multiple times.
                            # For efficiency, let's just break if we need to shrink.
                            # The simplex is likely stuck in a local minimum or flat region.
                            pass 
                        
                        # If we break here, we save budget for other starts.
                        break

        # --- Phase 3: Final Polish ---
        # If we have budget left, do a high-precision polish from the global best.
        if not check_deadline() and used < total_budget:
            x_polish = best_x.copy()
            f_polish = best_f
            sigma = 0.1 # Start with a small step
            
            # Use a simple (1+1)-ES for polishing
            while used < total_budget and not check_deadline():
                y = x_polish + sigma * rng.standard_normal(dim)
                y_clipped = np.clip(y, -5, 5)
                f_y = func(y_clipped)
                used += 1
                
                if f_y < f_polish:
                    x_polish = y_clipped
                    f_polish = f_y
                    if f_polish < best_f:
                        best_f = f_polish
                        best_x = x_polish.copy()
                    sigma *= 1.1 # Increase step if successful
                else:
                    sigma *= 0.9 # Decrease step if failed
                    if sigma < 1e-6:
                        break

        return best_x, best_f
