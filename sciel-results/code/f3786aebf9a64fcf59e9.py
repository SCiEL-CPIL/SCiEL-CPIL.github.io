"""Grid-based decomposition multi-objective optimizer for RE37."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Grid-based decomposition:
    1. Coarse grid evaluation to map the objective landscape.
    2. Identify non-dominated points on the grid.
    3. Select diverse non-dominated points for local refinement.
    4. Local refinement via Nelder-Mead on promising regions.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # 1. Coarse Grid
        # We want a grid that fits in ~50% of the budget to leave room for refinement.
        # For dim=4, g=3 -> 81 points. g=4 -> 256 points (too many for 200 budget).
        # So g=3 is the standard choice.
        g = 3
        while (g ** dim) > budget * 0.5:
            g -= 1
            if g < 2:
                g = 2
                break
        
        axes = [np.linspace(lo[i], hi[i], g) for i in range(dim)]
        grid = np.array(np.meshgrid(*axes)).reshape(dim, -1).T
        
        F_grid = np.array([problem(x) for x in grid])
        used = len(grid)
        
        # 2. Identify Non-Dominated Points
        # A point i is dominated if there exists j such that F[j] <= F[i] and F[j] < F[i]
        # Vectorized dominance check
        # F_grid shape: (N, M)
        # less_equal: (N, N, M) -> all along M -> (N, N)
        less_equal = np.all(F_grid[:, np.newaxis, :] <= F_grid[np.newaxis, :, :], axis=2)
        less_than = np.any(F_grid[:, np.newaxis, :] < F_grid[np.newaxis, :, :], axis=2)
        dominates = less_equal & less_than
        
        # A point is non-dominated if no other point dominates it
        # dominates[i, j] is True if j dominates i
        is_non_dominated = ~np.any(dominates, axis=1)
        
        non_dom_indices = np.where(is_non_dominated)[0]
        
        if len(non_dom_indices) == 0:
            # Fallback: pick the point with the lowest sum of objectives
            non_dom_indices = np.array([np.argmin(np.sum(F_grid, axis=1))])
            
        # 3. Select Diverse Points for Refinement
        # We want to cover the front, not just the best knee.
        # Calculate remaining budget
        remaining = budget - used
        
        # Estimate cost per refinement: ~15-20 evals
        # We want to use as many as possible, but ensure at least 10 evals each.
        max_refine = max(1, remaining // 15)
        
        # If we have more non-dominated points than we can refine, select diverse ones.
        if len(non_dom_indices) > max_refine:
            # Greedy farthest point sampling in objective space
            selected = [non_dom_indices[0]]
            selected_F = [F_grid[non_dom_indices[0]]]
            
            while len(selected) < max_refine:
                # Calculate min distance from each candidate to selected set
                # candidates: non_dom_indices not in selected
                candidates = [idx for idx in non_dom_indices if idx not in selected]
                
                if not candidates:
                    break
                    
                best_idx = -1
                best_min_dist = -1
                
                for idx in candidates:
                    # Distance to each selected point
                    dists = [np.linalg.norm(F_grid[idx] - sf) for sf in selected_F]
                    min_dist = min(dists)
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_idx = idx
                        
                if best_idx == -1:
                    break
                selected.append(best_idx)
                selected_F.append(F_grid[best_idx])
            
            refine_indices = selected
        else:
            refine_indices = list(non_dom_indices)
            
        # 4. Local Refinement
        # Use Nelder-Mead for local optimization
        # We will use weighted sum scalarization with weights that emphasize different objectives
        # to guide the search towards different parts of the front if needed, 
        # but since we are starting from diverse points, simple Nelder-Mead on the hypervolume 
        # or a weighted sum is fine. 
        # Actually, Nelder-Mead minimizes a scalar function. 
        # We can use a weighted sum with weights based on the position of the point 
        # or just uniform weights. 
        # Uniform weights (1, 1, 1) will drive towards the knee.
        # To cover the front, we might want to bias the weights.
        # However, starting from diverse points on the front and minimizing a scalar 
        # like sum of objectives will tend to pull them towards the knee.
        # A better scalarization for local refinement on a known front is to use 
        # the ideal point and reference point to define a direction, or simply 
        # minimize the distance to the ideal point? No, that's also knee-biased.
        
        # Let's use a simple approach: Minimize the sum of objectives, but 
        # initialize with the diverse points. The local minimum of the sum 
        # near a diverse point might stay close if the front is convex enough.
        # Or, we can use a weighted sum where weights are inversely proportional 
        # to the objective values? 
        
        # Let's stick to a robust Nelder-Mead with tight bounds and limited iterations.
        # We will use the "augmented" objective: sum of objectives.
        
        for idx in refine_indices:
            if used >= budget:
                break
            
            x0 = grid[idx]
            
            # Define objective function
            def obj(x):
                nonlocal used
                if used >= budget:
                    return 1e9
                fx = problem(x)
                used += 1
                # Use a weighted sum. 
                # To encourage spreading, we could use a penalty if we get too close 
                # to other recorded points, but that's complex.
                # Let's just minimize the sum.
                return np.sum(fx)
            
            # Nelder-Mead options
            # maxfev: limit evaluations per refinement
            # We have `remaining` budget left. 
            # Let's allocate a fair share to each refinement.
            num_refinements_left = len(refine_indices) - refine_indices.index(idx)
            if num_refinements_left > 0:
                alloc_per_refine = (budget - used) // num_refinements_left
            else:
                alloc_per_refine = 0
                
            maxfev = min(20, alloc_per_refine)
            if maxfev <= 0:
                continue
                
            options = {
                'maxiter': 50,
                'maxfev': maxfev,
                'xatol': 1e-4,
                'fatol': 1e-4
            }
            
            try:
                res = minimize(obj, x0, method='Nelder-Mead', bounds=list(zip(lo, hi)), options=options)
                # Ensure the final point is evaluated and recorded
                # minimize might stop before hitting maxfev if it converges.
                # We want to record the best point found.
                # res.x is the best point.
                # We should evaluate it to ensure it's in the record?
                # The problem function is called inside obj, so it's already recorded.
                # However, if minimize returns a point that wasn't the last evaluation,
                # it's still in the history.
                # But to be safe, if we have budget left, we can explicitly evaluate res.x?
                # No, that might be redundant. The evaluator records EVERY returned vector.
                # So any call to problem(x) is recorded.
                
            except Exception:
                pass
                
        return None
# EVOLVE-BLOCK-END
