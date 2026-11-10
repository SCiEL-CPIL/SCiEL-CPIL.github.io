import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        
        # Strategy:
        # 1. Hypothesis: For WFG9, the Pareto front is determined by x[0] and x[1].
        #    The remaining variables x[2:] affect the distance to the front.
        #    Setting x[2:] to mid-range often minimizes the distance term h.
        # 2. Perform a dense grid search on x[0], x[1] with x[2:] fixed at mid.
        # 3. Use remaining budget for local refinement of non-dominated points.
        
        # Determine grid size based on budget
        # We need to leave some budget for refinement.
        # Let's aim for ~70% of budget for grid, 30% for refinement.
        grid_budget = int(self.budget * 0.7)
        
        # Grid resolution
        # sqrt(grid_budget) points per dimension
        n_grid = int(np.sqrt(grid_budget))
        if n_grid < 5:
            n_grid = 5
            
        # Generate grid for x[0] and x[1]
        # We sample the bounds for x[0] and x[1]
        x0_vals = np.linspace(lo[0], hi[0], n_grid)
        x1_vals = np.linspace(lo[1], hi[1], n_grid)
        
        X = []
        F = []
        
        # Fill the grid
        for x0 in x0_vals:
            for x1 in x1_vals:
                if time.time() > deadline:
                    break
                if len(X) >= self.budget:
                    break
                
                x = np.zeros(self.dim)
                x[0] = x0
                x[1] = x1
                # Fix remaining variables to mid-point
                # This is a heuristic based on WFG problem structure where 
                # tail variables contribute to distance-from-front.
                x[2:] = mid[2:]
                
                fx = problem(x)
                X.append(x)
                F.append(fx)
                
            if time.time() > deadline:
                break
            if len(X) >= self.budget:
                break
        
        if not X:
            # Fallback if grid failed or budget was tiny
            x = np.random.rand(self.dim) * (hi - lo) + lo
            fx = problem(x)
            return [x]

        X = np.array(X)
        F = np.array(F)
        
        # Local Refinement
        # Identify non-dominated points
        # A point i is dominated by j if F[j] <= F[i] and F[j] != F[i]
        
        def is_dominated(f, F_all):
            # Check if f is dominated by any point in F_all
            # f is dominated if exists g in F_all such that g <= f and g != f
            # For efficiency, we can just check against the current set of candidates
            # But here we just want to pick the best ones for refinement.
            # We'll just pick points that are not dominated by others in the current set.
            pass

        # Simple non-dominated sorting for selection
        # Since n is small (~100-140), O(N^2) is fine.
        dominated = np.zeros(len(F), dtype=bool)
        for i in range(len(F)):
            for j in range(len(F)):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        
        # Select non-dominated points for refinement
        # If too many, select a diverse subset or just the first few.
        # We have remaining_budget = self.budget - len(X)
        remaining_budget = self.budget - len(X)
        
        if remaining_budget > 0:
            # Get indices of non-dominated points
            nd_indices = np.where(~dominated)[0]
            
            # If we have more nd points than budget, we can't refine all.
            # Let's refine the ones with the lowest sum of objectives (proxy for closeness to ideal)
            # Or just pick a random subset if too many.
            if len(nd_indices) > remaining_budget:
                # Sort by sum of objectives
                sums = np.sum(F[nd_indices], axis=1)
                order = np.argsort(sums)
                nd_indices = nd_indices[order[:remaining_budget]]
            
            # Local search: Simple random perturbation + acceptance if better
            # Or a small Nelder-Mead? 
            # Given the time limit, a simple iterative improvement is safer.
            
            for idx in nd_indices:
                if time.time() > deadline:
                    break
                if remaining_budget <= 0:
                    break
                    
                x_curr = X[idx].copy()
                f_curr = F[idx].copy()
                
                # Try a few local steps
                # Use a small step size
                step_size = 0.05 * (hi - lo)
                
                # We will use the remaining budget for this point? 
                # No, we should distribute. 
                # Let's do 2-3 steps per point if budget allows.
                # Actually, let's just do 1 step per point if we have many points.
                # If we have few points, do more steps.
                
                n_steps = min(remaining_budget, 3) # Max 3 steps per point
                if len(nd_indices) > 1:
                    n_steps = 1 # If many points, only 1 step each
                
                for _ in range(n_steps):
                    if remaining_budget <= 0 or time.time() > deadline:
                        break
                    
                    # Random perturbation
                    delta = np.random.randn(self.dim) * step_size
                    x_new = np.clip(x_curr + delta, lo, hi)
                    f_new = problem(x_new)
                    remaining_budget -= 1
                    
                    # Accept if non-dominated by current best? 
                    # Simple: Accept if it improves the sum, or is non-dominated by f_curr
                    if np.any(f_new < f_curr):
                        # Check if f_new is dominated by f_curr? No, if f_new < f_curr in any obj, it might be better.
                        # But we want to keep the Pareto front.
                        # Let's replace x_curr, f_curr with x_new, f_new if f_new is not dominated by f_curr
                        # Actually, if f_new is better in one obj and worse in another, it's a trade-off.
                        # We should keep both if possible. But we are updating the slot.
                        # Let's just accept if the sum is lower (scalarization proxy)
                        if np.sum(f_new) < np.sum(f_curr):
                            x_curr = x_new
                            f_curr = f_new
                            
                X[idx] = x_curr
                F[idx] = f_curr
        
        return X
