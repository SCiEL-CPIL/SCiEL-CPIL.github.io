import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0
        dim = self.dim
        n_obj = self.n_obj
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # 1. Initial Sampling: Latin Hypercube
        n_init = 50
        X_init = np.empty((n_init, dim))
        for i in range(dim):
            # LHS: divide [0,1] into n_init intervals, pick one random point in each
            X_init[:, i] = lo[i] + span[i] * (np.random.rand(n_init) + np.arange(n_init)) / n_init
        idx = np.random.permutation(n_init)
        X_init = X_init[idx]
        
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
        X_all = X_init.copy()
        F_all = F_init.copy()
        
        # Helper to get non-dominated indices
        def get_non_dominated(F):
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True
            return np.where(~dominated)[0]
        
        # Local Search Loop
        # We will perform multiple rounds of local search
        # Round 1: 5 neighbors per non-dominated point
        # Round 2: 5 neighbors per non-dominated point
        # Round 3: 5 neighbors per non-dominated point
        
        rounds = 3
        neighbors_per_point = 5
        radius_factor = 0.1 # 10% of the span
        
        for r in range(rounds):
            if used >= self.budget or time.time() > deadline:
                break
            
            nd_idx = get_non_dominated(F_all)
            if len(nd_idx) == 0:
                break
                
            # To limit evaluations, we might not want to search every single non-dominated point
            # if there are too many. But with 50 initial, nd set is usually small (<20).
            # Let's cap the number of points to refine per round to ensure we finish.
            max_refine = min(len(nd_idx), 15)
            # Select the 'best' non-dominated points to refine?
            # Or just take the first max_refine.
            # Better: select points that are 'spread out' or just random selection from nd_idx
            # Random selection helps avoid bias.
            refine_idx = np.random.choice(nd_idx, max_refine, replace=False)
            
            for idx in refine_idx:
                if used >= self.budget or time.time() > deadline:
                    break
                
                base_point = X_all[idx]
                # Generate neighbors
                for k in range(neighbors_per_point):
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    # Perturbation: Gaussian noise scaled by span and radius_factor
                    # We decrease radius in later rounds?
                    # Or keep it constant. Let's decrease it to polish.
                    current_radius = radius_factor * (0.5 ** r)
                    
                    noise = np.random.randn(dim) * current_radius * span
                    new_point = base_point + noise
                    new_point = np.clip(new_point, lo, hi)
                    
                    # Check if this point is too close to an existing point to avoid waste
                    # Simple check: distance to base point should be > epsilon
                    # But we already generated it.
                    
                    f_new = problem(new_point)
                    used += 1
                    
                    X_all = np.vstack([X_all, [new_point]])
                    F_all = np.vstack([F_all, [f_new]])
                    
        return X_all
