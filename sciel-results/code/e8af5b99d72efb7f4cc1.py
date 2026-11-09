"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for WFG9.
    Uses a deterministic grid over the first 3 variables (which directly determine
    the Pareto front) with remaining variables optimized, followed by a local
    search phase to refine the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0

        # WFG9: variables 0-2 shape the Pareto front; variables 3-11 affect distance.
        # Strategy:
        # 1. Coarse grid on vars 0-2 (g=4 -> 64 points), vars 3-11 at midpoint
        # 2. For each grid point, quickly optimize vars 3-11 by trying 0 and hi
        # 3. Refine the non-dominated set with local search on all 12 vars
        
        g = 4
        grid_points = []
        for i in range(g):
            for j in range(g):
                for k in range(g):
                    x = np.full(self.dim, 0.0)
                    x[0] = lo[0] + (hi[0] - lo[0]) * i / (g - 1)
                    x[1] = lo[1] + (hi[1] - lo[1]) * j / (g - 1)
                    x[2] = lo[2] + (hi[2] - lo[2]) * k / (g - 1)
                    for d in range(3, self.dim):
                        x[d] = mid[d]
                    grid_points.append(x)
        
        grid_points = np.array(grid_points)
        n_grid = len(grid_points)
        
        # Evaluate grid points with midpoint for vars 3-11
        F_grid = np.zeros((n_grid, self.n_obj))
        for i in range(n_grid):
            if time.time() > deadline:
                break
            F_grid[i] = problem(grid_points[i])
        
        used = n_grid
        
        # Phase 2: For each grid point, try setting vars 3-11 to lo and hi
        # This is cheap and often significantly improves objectives
        # We'll do this for a subset if budget is tight
        # Budget remaining: 200 - 64 = 136
        # Each grid point can be tested with vars 3-11 at lo (1 eval) and hi (1 eval)
        # That's 64 * 2 = 128 evals, leaving 8 for refinement
        # Let's do lo for all, and hi for a few
        
        X_all = [grid_points[i].copy() for i in range(n_grid)]
        F_all = [F_grid[i].copy() for i in range(n_grid)]
        
        # Helper: get non-dominated indices (vectorized-ish)
        def get_nondominated_indices(F_arr):
            n = len(F_arr)
            if n == 0:
                return []
            # F_arr is (n, n_obj)
            # Point i is dominated if exists j: F[j] <= F[i] and F[j] != F[i]
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
            return [i for i in range(n) if not dominated[i]]
        
        # Try vars 3-11 at lower bound for all grid points
        for i in range(n_grid):
            if used >= self.budget or time.time() > deadline:
                break
            x_lo = X_all[i].copy()
            for d in range(3, self.dim):
                x_lo[d] = lo[d]
            f_lo = problem(x_lo)
            used += 1
            
            # Check if this improves
            if np.all(f_lo <= F_all[i]):
                if not (np.all(f_lo == F_all[i])):
                    X_all[i] = x_lo
                    F_all[i] = f_lo
        
        # Try vars 3-11 at upper bound for remaining budget
        if used < self.budget and time.time() < deadline:
            for i in range(n_grid):
                if used >= self.budget or time.time() > deadline:
                    break
                x_hi = X_all[i].copy()
                for d in range(3, self.dim):
                    x_hi[d] = hi[d]
                f_hi = problem(x_hi)
                used += 1
                
                # Check if this improves
                if np.all(f_hi <= F_all[i]):
                    if not (np.all(f_hi == F_all[i])):
                        X_all[i] = x_hi
                        F_all[i] = f_hi
        
        # Phase 3: Local search on non-dominated points
        # Use smaller steps, focus on shape variables (0-2) with fine steps
        # and distance variables (3-11) with steps toward optimal bounds
        
        def dominates(f_new, f_old):
            return np.all(f_new <= f_old) and np.any(f_new < f_old)
        
        def is_dominated_by_any(f_new, F_archive):
            for f in F_archive:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False
        
        # Local search: for each non-dominated point, try small perturbations
        # on shape variables (0-2)
        remaining_budget = self.budget - used
        # Distribute remaining budget among non-dominated points
        nondom_idx = get_nondominated_indices(np.array(F_all))
        
        if nondom_idx and remaining_budget > 0:
            # Sort non-dominated points by their "spread" to prioritize
            # Just cycle through them with small steps
            n_nondom = len(nondom_idx)
            steps_per_point = max(1, remaining_budget // max(1, n_nondom))
            
            # Use a decaying step size
            base_step = 0.05 * (hi[:3] - lo[:3])
            
            for iteration in range(steps_per_point):
                if used >= self.budget or time.time() > deadline:
                    break
                
                # Get current non-dominated set
                nondom_idx = get_nondominated_indices(np.array(F_all))
                if not nondom_idx:
                    break
                
                for idx in nondom_idx:
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    x = X_all[idx].copy()
                    f = F_all[idx].copy()
                    
                    # Try perturbing each shape variable in both directions
                    for d in range(3):
                        if used >= self.budget or time.time() > deadline:
                            break
                        
                        step = base_step[d] / (1 + 0.5 * iteration)
                        
                        for direction in [-1, 1]:
                            if used >= self.budget or time.time() > deadline:
                                break
                            
                            x_new = x.copy()
                            x_new[d] += direction * step
                            x_new[d] = np.clip(x_new[d], lo[d], hi[d])
                            
                            f_new = problem(x_new)
                            used += 1
                            
                            if dominates(f_new, f):
                                X_all[idx] = x_new
                                F_all[idx] = f_new
                                # Restart inner loop with new point
                                break
                            elif not is_dominated_by_any(f_new, F_all):
                                # Add as new non-dominated point
                                X_all.append(x_new)
                                F_all.append(f_new)
                                break  # Move to next shape variable
        
        # Phase 4: Final polish - very fine steps on the final non-dominated set
        if used < self.budget and time.time() < deadline:
            nondom_idx = get_nondominated_indices(np.array(F_all))
            if nondom_idx:
                # Sort by objective sum for prioritization
                sums = np.sum(np.array(F_all), axis=1)
                sorted_idx = sorted(range(len(F_all)), key=lambda i: sums[i])
                
                # Only polish the top candidates
                candidates = [i for i in sorted_idx if i in set(nondom_idx)][:min(10, len(nondom_idx))]
                
                fine_step = 0.01 * (hi[:3] - lo[:3])
                
                for idx in candidates:
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    x = X_all[idx].copy()
                    f = F_all[idx].copy()
                    
                    for d in range(3):
                        if used >= self.budget or time.time() > deadline:
                            break
                        
                        for direction in [-1, 1]:
                            if used >= self.budget or time.time() > deadline:
                                break
                            
                            x_new = x.copy()
                            x_new[d] += direction * fine_step
                            x_new[d] = np.clip(x_new[d], lo[d], hi[d])
                            
                            f_new = problem(x_new)
                            used += 1
                            
                            if dominates(f_new, f):
                                X_all[idx] = x_new
                                F_all[idx] = f_new
                                break
                            elif not is_dominated_by_any(f_new, F_all):
                                X_all.append(x_new)
                                F_all.append(f_new)
                                break
        
        # Return all evaluated points (the evaluator records all objective vectors)
        final_F = np.array(F_all)
        final_X = np.array(X_all)
        
        return final_X
# EVOLVE-BLOCK-END
