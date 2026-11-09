import numpy as np
import time
import itertools

class Optimizer:
    """
    Optimizer for WFG9 using a structured pair-sweep and local polishing strategy.
    Exploits the likely structure where trade-offs are determined by subsets of variables,
    and other variables act as noise/deviation that can be locally minimized.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = self.start_time + 290.0  # 300s limit, 10s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        
        # Archive to store non-dominated points
        archive_X = []
        archive_F = []
        
        def is_dominated(f_new, archive_F):
            if not archive_F:
                return False
            # Check if any point in archive dominates f_new
            # f_arch <= f_new and f_arch != f_new
            # Vectorized check
            # archive_F shape (N, M), f_new shape (M,)
            # We need any row in archive_F that is <= f_new and < f_new in at least one dim
            leq = np.all(archive_F <= f_new, axis=1)
            lt = np.any(archive_F < f_new, axis=1)
            return np.any(leq & lt)

        def add_point(x, f):
            # Check if x is dominated by any in archive
            if is_dominated(f, archive_F):
                return False
            
            # Remove points in archive that are dominated by f
            if archive_F:
                keep_mask = []
                for i, f_arch in enumerate(archive_F):
                    # If f dominates f_arch, remove f_arch
                    # f <= f_arch and f != f_arch
                    dom = np.all(f <= f_arch) and np.any(f < f_arch)
                    if not dom:
                        keep_mask.append(True)
                    else:
                        keep_mask.append(False)
                
                if not all(keep_mask):
                    # Filter archive
                    new_archive_X = [archive_X[i] for i in range(len(archive_X)) if keep_mask[i]]
                    new_archive_F = [archive_F[i] for i in range(len(archive_F)) if keep_mask[i]]
                    # Clear and re-add to avoid list mutation issues during iteration if we were careful, 
                    # but here we just rebuild
                    archive_X.clear()
                    archive_F.clear()
                    archive_X.extend(new_archive_X)
                    archive_F.extend(new_archive_F)
            
            archive_X.append(x)
            archive_F.append(f)
            return True

        def evaluate_and_add(x):
            if time.time() > deadline:
                return
            f = np.array(problem(x))
            add_point(x, f)

        # Phase 1: Structured Pair Sweep
        # We assume the first few variables are more likely to determine the Pareto front shape.
        # We will sweep pairs of indices.
        # To cover the space, we pick pairs from the first 6 variables, 
        # and also a few random pairs to be safe.
        
        indices = list(range(self.dim))
        # Prioritize low indices
        priority_pairs = list(itertools.combinations(range(min(6, self.dim)), 2))
        # Add some higher index pairs if budget allows
        remaining_indices = list(range(min(6, self.dim), self.dim))
        if len(remaining_indices) > 1:
            # Add a few pairs involving higher indices
            extra_pairs = list(itertools.combinations(remaining_indices, 2))[:2]
            priority_pairs.extend(extra_pairs)
        
        # Shuffle to ensure variety if budget is tight, but keep low indices first
        # Actually, let's just use the first 10 pairs from the priority list
        pairs_to_test = priority_pairs[:10]
        
        # Grid points per pair: 5x2 = 10 points? Or 3x3=9?
        # Let's use 5 values for the first var in pair, 2 for the second.
        # Total points per pair = 10.
        # 10 pairs * 10 points = 100 evaluations.
        
        grid_vals_1 = np.linspace(0.1, 0.9, 5)
        grid_vals_2 = np.linspace(0.25, 0.75, 2)
        
        used_evals = 0
        max_evals_phase1 = 100
        
        for i, j in pairs_to_test:
            if used_evals >= max_evals_phase1:
                break
            if time.time() > deadline:
                break
            
            # Create base vector: midpoints
            base_x = mid.copy()
            
            for v1 in grid_vals_1:
                for v2 in grid_vals_2:
                    if time.time() > deadline:
                        break
                    if used_evals >= self.budget:
                        break
                    
                    x = base_x.copy()
                    x[i] = lo[i] + v1 * (hi[i] - lo[i])
                    x[j] = lo[j] + v2 * (hi[j] - lo[j])
                    
                    evaluate_and_add(x)
                    used_evals += 1

        # Phase 2: Local Polish
        # Use remaining budget to refine the best points.
        # We select points from the archive that are well-spread.
        
        if archive_F:
            F_arr = np.array(archive_F)
            # Normalize objectives to [0,1] for distance calculation
            f_min = np.min(F_arr, axis=0)
            f_max = np.max(F_arr, axis=0)
            f_range = f_max - f_min
            f_range[f_range == 0] = 1.0
            F_norm = (F_arr - f_min) / f_range
            
            # Select top K points based on diversity (e.g., k-means++ or just random sample of non-dominated)
            # Simple approach: Pick 10 random points from the archive, or the ones with lowest sum of normalized objectives?
            # Better: Pick points that are far apart.
            # Let's just pick 10 random non-dominated points to polish.
            n_polish = min(10, len(archive_X))
            if n_polish > 0:
                # Indices of points to polish
                polish_indices = np.random.choice(len(archive_X), n_polish, replace=False)
                
                budget_left = self.budget - used_evals
                iters_per_point = max(1, budget_left // n_polish)
                
                for idx in polish_indices:
                    if time.time() > deadline:
                        break
                    if used_evals >= self.budget:
                        break
                    
                    x_cur = archive_X[idx].copy()
                    f_cur = archive_F[idx].copy()
                    
                    # Local Coordinate Descent on Sum of Objectives
                    # This tends to move points towards the Pareto Front (minimizing all objectives jointly)
                    step_size = 0.1
                    for _ in range(iters_per_point):
                        if time.time() > deadline:
                            break
                        if used_evals >= self.budget:
                            break
                        
                        best_x = x_cur
                        best_f = f_cur
                        best_sum = np.sum(f_cur)
                        
                        # Try moving in each dimension
                        for d in range(self.dim):
                            # Positive step
                            x_try = x_cur.copy()
                            x_try[d] += step_size * (hi[d] - lo[d])
                            if x_try[d] > hi[d]: x_try[d] = hi[d]
                            
                            f_try = np.array(problem(x_try))
                            used_evals += 1
                            
                            sum_try = np.sum(f_try)
                            # We want to minimize the sum, but also ensure we don't go backwards in a way that hurts diversity?
                            # For polishing, minimizing sum is a good proxy for finding the PF.
                            if sum_try < best_sum:
                                best_sum = sum_try
                                best_x = x_try
                                best_f = f_try
                            
                            # Negative step
                            x_try = x_cur.copy()
                            x_try[d] -= step_size * (hi[d] - lo[d])
                            if x_try[d] < lo[d]: x_try[d] = lo[d]
                            
                            f_try = np.array(problem(x_try))
                            used_evals += 1
                            
                            sum_try = np.sum(f_try)
                            if sum_try < best_sum:
                                best_sum = sum_try
                                best_x = x_try
                                best_f = f_try
                            
                            if time.time() > deadline:
                                break
                            if used_evals >= self.budget:
                                break
                        
                        # Update current point if improvement
                        if not np.allclose(x_cur, best_x):
                            x_cur = best_x
                            f_cur = best_f
                            add_point(x_cur, f_cur)
                            # Reduce step size for next iteration
                            step_size *= 0.5
                        else:
                            break # No improvement, stop polishing this point

        return archive_X
