"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for WFG9.
    Uses a 5x5x5 grid over the first 3 variables (which directly determine
    the Pareto front) with remaining variables at lower bound (0), followed by
    a local search phase to refine the front.
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
        
        # WFG9: the first 3 variables determine the Pareto front.
        # Variables 3-11 affect the distance to the front.
        # Setting them to lower bound (0) typically minimizes distance.
        
        g = 5
        grid_points = []
        for i in range(g):
            for j in range(g):
                for k in range(g):
                    x = np.full(self.dim, 0.0)
                    x[0] = lo[0] + (hi[0] - lo[0]) * i / (g - 1)
                    x[1] = lo[1] + (hi[1] - lo[1]) * j / (g - 1)
                    x[2] = lo[2] + (hi[2] - lo[2]) * k / (g - 1)
                    # Set remaining variables to lower bound (0)
                    for d in range(3, self.dim):
                        x[d] = lo[d]
                    grid_points.append(x)
        
        grid_points = np.array(grid_points)
        n_grid = len(grid_points)
        
        # Evaluate grid points
        F = np.zeros((n_grid, self.n_obj))
        for i in range(n_grid):
            if time.time() > deadline:
                break
            F[i] = problem(grid_points[i])
        
        used = n_grid
        
        # Keep track of all evaluated points
        X_all = [grid_points[i] for i in range(n_grid)]
        F_all = [F[i] for i in range(n_grid)]
        
        # Helper to check if a point is dominated by any in the archive
        def is_dominated_by_archive(f_new, F_archive, exclude_idx=None):
            for j, f in enumerate(F_archive):
                if j == exclude_idx:
                    continue
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False

        # Helper to check if f_new dominates f_old
        def dominates(f_new, f_old):
            return np.all(f_new <= f_old) and np.any(f_new < f_old)

        # Local search phase: refine non-dominated points
        # We will perform coordinate descent on the non-dominated points.
        # Focus on variables 3-11 to minimize distance, and 0-2 for fine tuning.
        
        while used < self.budget and time.time() < deadline:
            # Find non-dominated points
            nondom_idx = []
            for i in range(len(F_all)):
                dominated = False
                for j in range(len(F_all)):
                    if i != j:
                        if np.all(F_all[j] <= F_all[i]) and np.any(F_all[j] < F_all[i]):
                            dominated = True
                            break
                if not dominated:
                    nondom_idx.append(i)
            
            if not nondom_idx:
                break
            
            # Pick a non-dominated point to refine
            # We cycle through them to ensure all are improved
            # To be efficient, we pick the one with the highest "potential" (e.g., sum of objectives)
            # Or just cycle. Let's cycle for simplicity and fairness.
            # To avoid getting stuck in a local optimum, we can use a greedy approach:
            # Try all possible single-step improvements for all non-dominated points and pick the best one.
            
            best_idx = -1
            best_dim = -1
            best_direction = 0
            best_f_new = None
            best_x_new = None
            best_improvement = 0 # We want to minimize sum of objectives or maximize hypervolume contribution
            
            # Try small steps for each non-dominated point
            # Step size for shape vars (0-2)
            step_shape = 0.05 * (hi[:3] - lo[:3])
            # Step size for distance vars (3-11)
            step_dist = 0.1 * (hi[3:] - lo[3:])
            
            for idx in nondom_idx:
                x = X_all[idx]
                f = F_all[idx]
                
                # Try perturbing shape variables (0-2)
                for d in range(3):
                    for direction in [-1, 1]:
                        x_new = x.copy()
                        x_new[d] += direction * step_shape[d]
                        x_new[d] = np.clip(x_new[d], lo[d], hi[d])
                        
                        # Evaluate
                        f_new = problem(x_new)
                        used += 1
                        
                        if used >= self.budget or time.time() > deadline:
                            # If we run out of budget, we must return immediately
                            # But we need to record this evaluation
                            X_all.append(x_new)
                            F_all.append(f_new)
                            return np.array(X_all)
                        
                        # Check if it's an improvement (non-dominated and better than current)
                        if not is_dominated_by_archive(f_new, F_all, exclude_idx=idx):
                            # It's a candidate
                            # We prefer points that dominate others or are "better" in some sense
                            # For now, just track it
                            if best_f_new is None or np.sum(f_new) < np.sum(best_f_new):
                                best_idx = idx
                                best_dim = d
                                best_direction = direction
                                best_f_new = f_new
                                best_x_new = x_new
                            continue # Don't add to archive yet, wait to see if we find a better one in this batch?
                            # Actually, adding to archive immediately is safer for hypervolume.
                            # But we are in a loop. Let's add it if it's non-dominated.
                            # If we add it, it might dominate other candidates in this batch.
                            # Let's just add it if it's non-dominated.
                            X_all.append(x_new)
                            F_all.append(f_new)
                            # Remove dominated points from archive? No, keep all evaluated points.
                            # The evaluator records ALL points. We just use non-dominated for search.
                            # So we can just append.
                            best_idx = -1 # Reset best to avoid replacing?
                            # Actually, we want to keep the best improvement for the next iteration?
                            # No, we just want to make progress.
                            # Let's break out of the inner loops if we found a good improvement?
                            # No, let's just continue and let the next iteration of the while loop re-evaluate.
                            # But we've already spent the budget.
                            # Let's just break the inner loops to save time if we found a good one?
                            # No, we need to use the budget.
                            # Let's just continue.
                
                # Try perturbing distance variables (3-11)
                for d in range(3, self.dim):
                    # Bias towards lower bound (0)
                    if x[d] > lo[d]:
                        direction = -1
                    else:
                        direction = 1
                    
                    x_new = x.copy()
                    x_new[d] += direction * step_dist[d-3]
                    x_new[d] = np.clip(x_new[d], lo[d], hi[d])
                    
                    f_new = problem(x_new)
                    used += 1
                    
                    if used >= self.budget or time.time() > deadline:
                        X_all.append(x_new)
                        F_all.append(f_new)
                        return np.array(X_all)
                    
                    if not is_dominated_by_archive(f_new, F_all, exclude_idx=idx):
                        X_all.append(x_new)
                        F_all.append(f_new)
        
        # Return all evaluated points
        final_X = np.array(X_all)
        return final_X
# EVOLVE-BLOCK-END
