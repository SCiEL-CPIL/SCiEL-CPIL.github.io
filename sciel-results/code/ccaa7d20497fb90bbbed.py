"""Grid-based decomposition multi-objective optimizer for RE37."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Grid-based decomposition:
    1. Coarse grid evaluation to map the objective landscape.
    2. Identify non-dominated points on the grid.
    3. Local refinement via Nelder-Mead on promising regions.
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

        # Budget allocation:
        # - Use ~60% of budget for coarse grid
        # - Use ~40% for local refinement
        # Grid size: we want a manageable number of grid points.
        # For 4D, if we use g points per dimension, total = g^4.
        # Let's aim for grid_points <= budget * 0.6
        # g^4 <= 0.6 * 200 = 120 -> g <= 3.3 -> g=3 gives 81 points.
        # g=4 gives 256 points, which is too many.
        # Let's try g=3 (81 points) for coarse grid.
        # Remaining budget: 200 - 81 = 119 for refinement.
        
        # Actually, let's be more dynamic.
        # If budget is large, we can afford a finer grid.
        # For dim=4, g=3 is 81, g=4 is 256.
        # Let's stick with g=3 for the coarse grid to be safe.
        
        g = 3
        grid_points_per_dim = g
        total_grid_points = grid_points_per_dim ** dim
        
        # If total_grid_points > budget * 0.5, reduce g
        while total_grid_points > budget * 0.5 and grid_points_per_dim > 2:
            grid_points_per_dim -= 1
            total_grid_points = grid_points_per_dim ** dim
        
        # Generate grid points
        axes = [np.linspace(lo[i], hi[i], grid_points_per_dim) for i in range(dim)]
        grid = np.array(np.meshgrid(*axes)).reshape(dim, -1).T
        
        # Evaluate grid points
        F_grid = np.array([problem(x) for x in grid])
        used = total_grid_points
        
        # Find non-dominated points in the grid
        def is_dominated(f, F):
            # f is dominated if there exists f' in F such that f' <= f and f' < f
            # Vectorized check
            # For each f in F, check if f <= f and f < f
            # We want to find if any f' in F dominates f
            # This is O(N^2) which is fine for small N
            dominated = np.zeros(len(F), dtype=bool)
            for i, fi in enumerate(F):
                # Check if any fj (j != i) dominates fi
                # fj <= fi and fj < fi
                less_equal = np.all(F <= fi, axis=1)
                less_than = np.any(F < fi, axis=1)
                dominates = less_equal & less_than
                dominates[i] = False  # f does not dominate itself
                if np.any(dominates):
                    dominated[i] = True
            return dominated
        
        dom_grid = is_dominated(None, F_grid)
        non_dom_indices = np.where(~dom_grid)[0]
        
        # If no non-dominated points (shouldn't happen), take all
        if len(non_dom_indices) == 0:
            non_dom_indices = np.arange(len(grid))
        
        # Select a subset of non-dominated points for refinement
        # We want to use the remaining budget for Nelder-Mead.
        # Nelder-Mead in 4D takes roughly 10-20 evaluations per run.
        # Let's say we can afford ~5-10 refinements.
        # Pick the "most central" non-dominated points or spread them out.
        # Simple strategy: pick up to 10 non-dominated points, evenly spaced.
        
        remaining_budget = budget - used
        # Estimate cost per refinement: ~15 evaluations
        n_refine = min(10, len(non_dom_indices), max(1, remaining_budget // 15))
        
        if n_refine > 0:
            # Select n_refine points from non_dom_indices
            # Use a simple stratified selection to cover the front
            step = len(non_dom_indices) // n_refine
            selected_indices = [non_dom_indices[i * step] for i in range(n_refine)]
            # Ensure we don't go out of bounds
            selected_indices = selected_indices[:n_refine]
            
            X_best = []
            F_best = []
            
            for idx in selected_indices:
                if used >= budget:
                    break
                x0 = grid[idx]
                f0 = F_grid[idx]
                
                # Define objective for Nelder-Mead: sum of objectives (scalarization)
                # This is a simple heuristic; better would be to use a reference point
                def obj(x):
                    nonlocal used
                    if used >= budget:
                        return np.inf
                    fx = problem(x)
                    used += 1
                    # Use a weighted sum with equal weights, or just sum
                    # Since we want to minimize all, sum is a reasonable scalarization
                    return np.sum(fx)
                
                # Nelder-Mead options
                options = {'maxiter': 50, 'maxfev': max(1, budget - used)}
                
                try:
                    res = minimize(obj, x0, method='Nelder-Mead', bounds=list(zip(lo, hi)), options=options)
                    if used < budget:
                        xf = res.x
                        ff = problem(xf)
                        used += 1
                        X_best.append(xf)
                        F_best.append(ff)
                except Exception:
                    pass
        
        # Combine grid points and refined points
        # Return the set of all evaluated points (or just the best ones)
        # The problem statement says "Every returned objective vector is recorded"
        # and the return value of __call__ is ignored.
        # So we just need to ensure we've made the evaluations.
        
        return grid  # Return something valid
# EVOLVE-BLOCK-END
