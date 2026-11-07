# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Latin Hypercube Sampling
        # Use a portion of the budget for broad exploration
        n_lhs = int(budget * 0.6)
        if n_lhs < 10:
            n_lhs = 10
            
        # Generate LHS points
        # For each dimension, divide the range into n_lhs equal intervals
        # and pick one point uniformly at random from each interval
        lhs_points = np.zeros((n_lhs, dim))
        for i in range(dim):
            perm = np.random.permutation(n_lhs)
            for k in range(n_lhs):
                # Interval for this permutation index
                start = perm[k] / n_lhs
                end = (perm[k] + 1) / n_lhs
                # Random point in interval
                lhs_points[k, i] = lo[i] + (hi[i] - lo[i]) * (start + np.random.random() * (end - start))
                
        X = lhs_points
        F = np.array([problem(x) for x in X])
        used = n_lhs
        
        # Helper to get non-dominated points
        def get_nondominated(F):
            if len(F) == 0:
                return np.array([])
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        # Phase 2: Local refinement / Evolutionary step
        # Use remaining budget to refine the front
        while used < budget:
            if time.time() > deadline - 10: # 10s safety margin
                break
                
            # Get current non-dominated set
            nd_idx = get_nondominated(F)
            
            if len(nd_idx) == 0:
                # Fallback if all dominated (should not happen if we keep best)
                parent_idx = 0
            else:
                # Pick a parent from non-dominated set
                # We can use a simple random selection or niche selection
                # Random selection is fine for low budget
                parent_local_idx = np.random.randint(len(nd_idx))
                parent_idx = nd_idx[parent_local_idx]
                
            parent = X[parent_idx]
            
            # Mutation: Gaussian noise
            # Scale noise based on distance to bounds or a fixed fraction
            # A common choice is 0.1 * range
            sigma = 0.1 * (hi - lo)
            child = parent + np.random.normal(0, sigma, dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Add to population
            X = np.vstack([X, child])
            F = np.vstack([F, fc])
            
            # Optional: Prune dominated points to keep computation fast?
            # With 200 points, O(N^2) dominance check is fine (200^2 = 40000 ops)
            
        return X
# EVOLVE-BLOCK-END
