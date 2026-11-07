# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using constrained random search in shrinking spheres for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # Deadline guard: 300s limit, use 290s to be safe
        start_time = time.time()
        deadline = start_time + 290
        
        # WFG9 has its Pareto front near the origin (lower bounds)
        # Use a shrinking sphere centered at lo
        center = lo.copy()
        
        # Phase 1: Initial broad sampling (LHS) to get a sense of the landscape
        n_init = min(50, n)
        lhs_points = np.zeros((n_init, dim))
        for d in range(dim):
            points = (np.arange(n_init) + np.random.random(n_init)) / n_init
            lhs_points[:, d] = points
            lhs_points[:, d] = lhs_points[np.random.permutation(n_init), d]
        
        X_init = lo + lhs_points * (hi - lo)
        F_init = np.array([problem(x) for x in X_init])
        
        # Track all evaluated points and their objectives for hypervolume
        all_x = X_init.copy()
        all_f = F_init.copy()
        evals_used = n_init
        
        # Phase 2: Shrinking sphere search centered at lo
        # WFG9's Pareto front is known to be near the origin
        # Use a geometrically decreasing radius
        
        n_remaining = n - evals_used
        if n_remaining <= 0:
            return all_x
        
        # Number of points per sphere level
        n_levels = 8
        points_per_level = max(1, n_remaining // n_levels)
        
        # Radii: start large (50% of range), shrink to small (1%)
        max_radius = 0.5 * np.max(hi - lo)
        min_radius = 0.01 * np.min(hi - lo)
        radius_ratios = np.geomspace(max_radius, min_radius, n_levels)
        
        for level in range(n_levels):
            if evals_used >= n:
                break
            
            radius = radius_ratios[level]
            n_points_this_level = min(points_per_level, n - evals_used)
            
            for _ in range(n_points_this_level):
                # Check deadline
                if time.time() > deadline:
                    break
                
                # Generate random point in sphere of given radius around center
                # Use Gaussian distribution scaled to radius
                direction = np.random.standard_normal(dim)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                
                # Random radius within [0, radius]
                r = radius * np.random.random()
                
                x_candidate = center + r * direction
                x_candidate = np.clip(x_candidate, lo, hi)
                
                f_candidate = problem(x_candidate)
                evals_used += 1
                
                # Add to archive
                all_x = np.vstack([all_x, x_candidate.reshape(1, -1)])
                all_f = np.vstack([all_f, f_candidate.reshape(1, -1)])
            
            if time.time() > deadline:
                break
        
        # Phase 3: Fine-tuning around best points found
        if evals_used < n and time.time() < deadline:
            # Find the point with minimum sum of objectives (simple heuristic)
            if len(all_f) > 0:
                obj_sums = np.sum(all_f, axis=1)
                best_idx = np.argmin(obj_sums)
                best_x = all_x[best_idx]
                
                n_fine = n - evals_used
                for _ in range(n_fine):
                    if time.time() > deadline:
                        break
                    
                    # Small perturbations around best point
                    sigma = 0.05 * (hi - lo)
                    x_fine = best_x + sigma * np.random.standard_normal(dim)
                    x_fine = np.clip(x_fine, lo, hi)
                    
                    f_fine = problem(x_fine)
                    evals_used += 1
                    
                    all_x = np.vstack([all_x, x_fine.reshape(1, -1)])
                    all_f = np.vstack([all_f, f_fine.reshape(1, -1)])
        
        return all_x
# EVOLVE-BLOCK-END
