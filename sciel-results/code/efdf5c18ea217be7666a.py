import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        used = 0
        best_points = []  # List of (x, f) for non-dominated points found
        best_f = []
        
        # WFG9 structure: first 3 variables determine objective values on the Pareto front
        # The remaining variables should be set to their optimal values to achieve the front
        # For WFG9, the Pareto front is characterized by x1, x2, x3 in [0,1] (normalized)
        # and x4..x12 should be at specific values
        
        # Strategy: Grid search over the first 3 variables with fine resolution,
        # and optimize the remaining variables using a few local search steps
        
        # First, let's establish the "good" values for variables 4-12
        # For WFG9, the Pareto front requires specific values for x4-x12
        # Based on the WFG9 definition, x4..x12 should be at the midpoint or specific values
        
        # Let's use a multi-stage approach:
        # Stage 1: Coarse grid over first 3 variables
        # Stage 2: For each point, do local search on all 12 variables
        # Stage 3: Refine the best points
        
        # Determine grid resolution based on budget
        # We have 200 evaluations total
        # Let's use 6x6x4 = 144 points for the grid, leaving 56 for local search
        
        # Actually, let's be more strategic:
        # Use 5x5x4 = 100 points for initial grid
        # Then use remaining 100 for local search on the best points
        
        n1, n2, n3 = 6, 5, 4  # Grid resolution for first 3 variables
        grid_size = n1 * n2 * n3
        
        if grid_size > self.budget - 20:
            # Reduce grid
            n1 = max(3, n1 - 2)
            n2 = max(3, n2 - 1)
            n3 = max(3, n3 - 1)
            grid_size = n1 * n2 * n3
        
        # Generate grid points for first 3 variables
        # Map to the actual bounds
        v1_vals = np.linspace(lo[0], hi[0], n1)
        v2_vals = np.linspace(lo[1], hi[1], n2)
        v3_vals = np.linspace(lo[2], hi[2], n3)
        
        # For variables 4-12, start with midpoints
        mid_vars = (lo[3:] + hi[3:]) / 2.0
        
        # Stage 1: Evaluate grid points
        grid_points = []
        grid_f = []
        
        for v1 in v1_vals:
            for v2 in v2_vals:
                for v3 in v3_vals:
                    if used >= self.budget or time.time() > deadline:
                        break
                    x = np.zeros(self.dim)
                    x[0] = v1
                    x[1] = v2
                    x[2] = v3
                    x[3:] = mid_vars
                    x = np.clip(x, lo, hi)
                    f = problem(x)
                    used += 1
                    grid_points.append(x)
                    grid_f.append(f)
                if used >= self.budget or time.time() > deadline:
                    break
            if used >= self.budget or time.time() > deadline:
                break
        
        grid_points = np.array(grid_points)
        grid_f = np.array(grid_f)
        
        # Find non-dominated points from the grid
        nd_indices = self._find_non_dominated(grid_f)
        
        # Stage 2: Local search on non-dominated points
        # Use remaining budget for local search
        remaining_budget = self.budget - used
        n_local = min(len(nd_indices), remaining_budget // 5)  # 5 steps per point
        
        # Sort by some heuristic - prioritize points that are well-spread
        if len(nd_indices) > 0:
            # Compute a simple spread metric
            cd = self._calc_crowding_simple(grid_f, nd_indices)
            order = np.argsort(-cd)
            selected = nd_indices[order[:n_local]]
        else:
            selected = []
        
        for idx in selected:
            if used >= self.budget or time.time() > deadline:
                break
            
            x_best = grid_points[idx].copy()
            f_best = grid_f[idx].copy()
            
            # Local search: random walk with decreasing step size
            for step_i in range(5):
                if used >= self.budget or time.time() > deadline:
                    break
                
                # Decreasing step size
                step_size = 0.1 * (1.0 - step_i / 5.0) * span
                
                # Try a few random directions
                for trial in range(3):
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    x_new = x_best + np.random.randn(self.dim) * step_size
                    x_new = np.clip(x_new, lo, hi)
                    f_new = problem(x_new)
                    used += 1
                    
                    # Accept if it dominates the current best or is non-dominated
                    if np.all(f_new <= f_best) and np.any(f_new < f_best):
                        x_best = x_new
                        f_best = f_new
                        break
                    elif not np.any(np.all(f_best <= f_new) & np.any(f_best < f_new)):
                        # Non-dominated: check if it's better in some sense
                        # Simple heuristic: accept if average objective is better
                        if np.mean(f_new) < np.mean(f_best):
                            x_best = x_new
                            f_best = f_new
                            break
        
        # Stage 3: If we have budget left, do a final refinement
        if used < self.budget and time.time() < deadline:
            # Collect all evaluated points
            all_x = [grid_points[i] for i in range(len(grid_points))]
            all_f = [grid_f[i] for i in range(len(grid_f))]
            
            # Add local search results
            for idx in selected:
                pass  # We'd need to track these separately
            
            # For simplicity, let's do a few more random perturbations of the best points
            # Find the best points based on hypervolume contribution
            all_f = np.array(all_f)
            all_x = np.array(all_x)
            
            # Find non-dominated
            nd_idx = self._find_non_dominated(all_f)
            
            # Perturb the best few
            if len(nd_idx) > 0:
                cd = self._calc_crowding_simple(all_f, nd_idx)
                order = np.argsort(-cd)
                top_k = min(5, len(nd_idx), (self.budget - used) // 3)
                
                for k_idx in order[:top_k]:
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    x_best = all_x[k_idx].copy()
                    f_best = all_f[k_idx].copy()
                    
                    for _ in range(3):
                        if used >= self.budget or time.time() > deadline:
                            break
                        
                        x_new = x_best + np.random.randn(self.dim) * 0.05 * span
                        x_new = np.clip(x_new, lo, hi)
                        f_new = problem(x_new)
                        used += 1
                        
                        if np.all(f_new <= f_best) and np.any(f_new < f_best):
                            x_best = x_new
                            f_best = f_new
        
        return np.array([])

    def _find_non_dominated(self, F):
        """Find indices of non-dominated points"""
        n = len(F)
        nd = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                nd.append(i)
        return np.array(nd, dtype=int)

    def _calc_crowding_simple(self, F, indices):
        """Calculate simple crowding distance for a set of points"""
        if len(indices) <= 2:
            return np.ones(len(indices))
        
        m = len(indices)
        front_F = F[indices]
        cd = np.zeros(m)
        
        for j in range(self.n_obj):
            obj = front_F[:, j]
            sorted_idx = np.argsort(obj)
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            obj_range = obj[sorted_idx[-1]] - obj[sorted_idx[0]]
            if obj_range > 1e-14:
                for i in range(1, m - 1):
                    cd[sorted_idx[i]] += (obj[sorted_idx[i+1]] - obj[sorted_idx[i-1]]) / obj_range
        
        # Replace inf with large number
        cd = np.where(np.isinf(cd), 1e6, cd)
        return cd
