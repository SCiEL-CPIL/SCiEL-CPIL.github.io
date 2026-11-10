import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 295.0
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        mid = 0.5 * (lo + hi)
        
        all_X = []
        all_F = []
        budget_used = 0
        
        def now():
            return time.time()
            
        def eval_point(x):
            nonlocal budget_used
            if budget_used >= budget or now() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            budget_used += 1
            all_X.append(x.copy())
            all_F.append(np.array(f))
            return f

        # Step 1: Determine optimal value for tail variables (indices 2 to dim-1)
        # Assume x[0] and x[1] are position variables.
        # Test tail values: lo, mid, hi
        best_tail_val = mid[2]
        best_tail_score = float('inf')
        
        # Base point for testing
        base_x = mid.copy()
        # Set x[0], x[1] to mid for testing
        base_x[0] = mid[0]
        base_x[1] = mid[1]
        
        candidates = [lo[2], mid[2], hi[2]]
        if dim > 2:
            for val in candidates:
                if budget_used >= budget or now() > deadline:
                    break
                test_x = base_x.copy()
                for d in range(2, dim):
                    test_x[d] = val
                f = eval_point(test_x)
                if f is not None:
                    # Use sum of objectives as a heuristic for "distance from front"
                    # If separable, minimizing the sum helps find the correct offset
                    score = np.sum(f)
                    if score < best_tail_score:
                        best_tail_score = score
                        best_tail_val = val
        
        # Fix tail variables to best_tail_val
        tail_val = best_tail_val
        
        # Step 2: Grid search on x[0] and x[1]
        # Remaining budget
        rem_budget = budget - budget_used
        
        # Allocate 90% to grid, 10% to refinement
        n_grid = int(np.sqrt(rem_budget * 0.9))
        if n_grid < 5:
            n_grid = 5
            
        x0_vals = np.linspace(lo[0], hi[0], n_grid)
        x1_vals = np.linspace(lo[1], hi[1], n_grid)
        
        for x0 in x0_vals:
            if budget_used >= budget or now() > deadline:
                break
            for x1 in x1_vals:
                if budget_used >= budget or now() > deadline:
                    break
                x = np.zeros(dim)
                x[0] = x0
                x[1] = x1
                for d in range(2, dim):
                    x[d] = tail_val
                eval_point(x)
                
        # Step 3: Refinement
        if all_F and budget_used < budget:
            F_arr = np.array(all_F)
            
            # Find non-dominated points
            non_dom_indices = []
            n_points = len(all_F)
            for i in range(n_points):
                f_i = F_arr[i]
                dominated = False
                for j in range(n_points):
                    if i != j:
                        f_j = F_arr[j]
                        if np.all(f_j <= f_i) and np.any(f_j < f_i):
                            dominated = True
                            break
                if not dominated:
                    non_dom_indices.append(i)
            
            # Refine around non-dominated points
            n_refine = min(budget - budget_used, 50)
            if n_refine > 0 and non_dom_indices:
                # Shuffle non-dom indices to diversify refinement
                np.random.shuffle(non_dom_indices)
                points_to_refine = non_dom_indices[:min(10, len(non_dom_indices))]
                
                evals_per_point = max(1, n_refine // max(1, len(points_to_refine)))
                
                for idx in points_to_refine:
                    if budget_used >= budget or now() > deadline:
                        break
                    
                    center_x = all_X[idx].copy()
                    
                    # Local search with decreasing step size
                    # Try small perturbations in x[0] and x[1]
                    # Also try slightly varying tail_val if it was at boundary? No, keep fixed.
                    
                    for _ in range(evals_per_point):
                        if budget_used >= budget or now() > deadline:
                            break
                        
                        x_new = center_x.copy()
                        # Gaussian perturbation on position vars
                        x_new[0] += np.random.normal(0, 0.05 * span[0])
                        x_new[1] += np.random.normal(0, 0.05 * span[1])
                        
                        # Keep tail vars fixed
                        for d in range(2, dim):
                            x_new[d] = tail_val
                            
                        eval_point(x_new)
                        
                        # If the new point is better (non-dominated), update center?
                        # Simple hill climbing: if new point dominates center, move center
                        # But we don't track dominance easily here in the loop.
                        # Just sample around the original center.
                        
        if all_X:
            return np.array(all_X)
        else:
            return np.empty((0, dim))
