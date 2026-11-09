"""Grid-based decomposition with Latin Hypercube Sampling for WFG4 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Grid-based decomposition with LHS and local refinement for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 330s kill

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # Phase 1: Latin Hypercube Sampling for broad coverage
        # Use a reasonable number of initial points
        n_lhs = min(120, self.budget // 2)
        
        # Generate LHS points
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            # Permute [0, 1, ..., n_lhs-1] / n_lhs to create stratified samples
            perm = rng.permutation(n_lhs)
            lhs_points[:, d] = (perm + rng.random(n_lhs)) / n_lhs
        
        X = lo + span * lhs_points
        F = np.array([problem(x) for x in X])
        used = n_lhs

        # Track all evaluated points and their objectives
        all_X = [X.copy()]
        all_F = [F.copy()]

        # Helper: check if we should stop
        def should_stop():
            return time.time() > deadline or used >= self.budget

        # Phase 2: Local refinement around best points
        # Identify non-dominated points
        def is_non_dominated(F_arr):
            """Return boolean mask of non-dominated points."""
            n = len(F_arr)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    # Check if j dominates i
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        mask[i] = False
                        break
            return mask

        # Refinement loop
        refinement_iter = 0
        while not should_stop() and refinement_iter < 50:
            # Combine all points
            combined_X = np.vstack(all_X)
            combined_F = np.vstack(all_F)
            
            # Find non-dominated points
            nd_mask = is_non_dominated(combined_F)
            nd_X = combined_X[nd_mask]
            nd_F = combined_F[nd_mask]
            
            if len(nd_X) == 0:
                break
            
            # Pick a random non-dominated point to refine
            idx = rng.integers(len(nd_X))
            center = nd_X[idx]
            
            # Generate several candidates around this center
            n_candidates = min(5, self.budget - used)
            if n_candidates <= 0:
                break
            
            # Adaptive step size based on span
            step_size = 0.05 * span
            
            candidates = center[None, :] + step_size * rng.standard_normal((n_candidates, dim))
            candidates = np.clip(candidates, lo, hi)
            
            new_X = []
            new_F = []
            for c in candidates:
                if used >= self.budget or time.time() > deadline:
                    break
                fc = problem(c)
                new_X.append(c)
                new_F.append(fc)
                used += 1
            
            if new_X:
                all_X.append(np.array(new_X))
                all_F.append(np.array(new_F))
            
            refinement_iter += 1

        # Phase 3: If we have budget left, do more LHS in a smaller region around best points
        if not should_stop():
            combined_X = np.vstack(all_X)
            combined_F = np.vstack(all_F)
            
            # Find the point with best sum of objectives (proxy for hypervolume contribution)
            # Normalize objectives to [0,1] range based on observed min/max
            obj_min = np.min(combined_F, axis=0)
            obj_max = np.max(combined_F, axis=0)
            obj_range = np.maximum(obj_max - obj_min, 1e-10)
            normalized_F = (combined_F - obj_min) / obj_range
            
            # Rank by sum of normalized objectives (lower is better)
            ranks = np.argsort(np.sum(normalized_F, axis=1))
            
            # Take top few points as centers for further refinement
            top_k = min(5, len(ranks))
            centers = combined_X[ranks[:top_k]]
            
            remaining_budget = self.budget - used
            if remaining_budget > 0:
                n_per_center = max(1, remaining_budget // top_k)
                
                for center in centers:
                    n_to_gen = min(n_per_center, self.budget - used)
                    if n_to_gen <= 0 or time.time() > deadline:
                        break
                    
                    # Generate candidates in a local neighborhood
                    local_step = 0.02 * span
                    local_candidates = center[None, :] + local_step * rng.standard_normal((n_to_gen, dim))
                    local_candidates = np.clip(local_candidates, lo, hi)
                    
                    new_X = []
                    new_F = []
                    for c in local_candidates:
                        if used >= self.budget or time.time() > deadline:
                            break
                        fc = problem(c)
                        new_X.append(c)
                        new_F.append(fc)
                        used += 1
                    
                    if new_X:
                        all_X.append(np.array(new_X))
                        all_F.append(np.array(new_F))

        # Return the combined set (evaluator ignores return value, but keep interface)
        if all_X:
            return np.vstack(all_X)
        return X
# EVOLVE-BLOCK-END
