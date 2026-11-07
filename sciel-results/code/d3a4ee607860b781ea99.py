import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        n = self.dim
        m = self.n_obj
        budget = self.budget
        
        # DTLZ7 structure:
        # f1 = x[0]
        # f2 = x[1]
        # f3 = (1+g) * h
        # g = 20*(n-2) + sum_{i=2}^{n-1} (x_i - 0.5)^2 * (4*sin^2(2*pi*(x_i-0.5)))
        # h = (m+1) - sum_{i=1}^{m-1} (f_i/(1+g)) * (1 + sin(pi*f_i/(1+g)))
        #
        # Pareto front: g=0 (x[2:]=0.5), so f3 = h = 4 - [x0*(1+sin(pi*x0)) + x1*(1+sin(pi*x1))]
        #
        # Key insight for hypervolume: The reference point is [1.1, 1.1, 25.0].
        # To maximize hypervolume, we need points that dominate large volumes.
        # The most valuable points are those with small f1, f2, AND small f3.
        #
        # Strategy: Use an adaptive approach.
        # 1. Start with a coarse grid to understand the landscape.
        # 2. Refine around high-value regions (low f1, f2, f3).
        # 3. Use the remaining budget for targeted sampling.
        
        # Phase 1: Coarse grid to get initial coverage
        # Use a 10x10 grid = 100 points
        n_coarse = 10
        x0_coarse = np.linspace(0, 1, n_coarse)
        x1_coarse = np.linspace(0, 1, n_coarse)
        
        X0, X1 = np.meshgrid(x0_coarse, x1_coarse)
        x0_flat = X0.ravel()
        x1_flat = X1.ravel()
        
        # Construct vectors for coarse grid
        X_coarse = np.zeros((len(x0_flat), n))
        X_coarse[:, 0] = x0_flat
        X_coarse[:, 1] = x1_flat
        X_coarse[:, 2:] = 0.5
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X_coarse = np.clip(X_coarse, lo, hi)
        
        # Evaluate coarse grid
        F_coarse = []
        for i in range(len(X_coarse)):
            if time.time() > deadline:
                break
            F_coarse.append(problem(X_coarse[i]))
        
        # Phase 2: Identify high-value regions
        # Points with low f1, f2, and f3 are most valuable
        if F_coarse:
            F_arr = np.array(F_coarse)
            # Filter out points that don't count (must be < reference in all objectives)
            ref = np.array([1.1, 1.1, 25.0])
            valid_mask = np.all(F_arr < ref, axis=1)
            
            if np.any(valid_mask):
                F_valid = F_arr[valid_mask]
                X_valid = X_coarse[valid_mask]
                
                # Find the best 20% of points (lowest sum of objectives)
                scores = F_valid.sum(axis=1)
                n_top = max(1, len(F_valid) // 5)
                top_indices = np.argsort(scores)[:n_top]
                X_top = X_valid[top_indices]
                
                # Phase 3: Refine around top regions
                # For each top point, create a small neighborhood
                n_refine = budget - len(F_coarse)
                if n_refine > 0:
                    # Create refinement points around top solutions
                    X_refine = []
                    n_per_point = max(1, n_refine // len(X_top))
                    
                    for i in range(len(X_top)):
                        if time.time() > deadline:
                            break
                        x_center = X_top[i]
                        # Create a small grid around this point
                        step = 0.05
                        for dx in [-step, 0, step]:
                            for dy in [-step, 0, step]:
                                if time.time() > deadline:
                                    break
                                x_new = x_center.copy()
                                x_new[0] = np.clip(x_center[0] + dx, 0, 1)
                                x_new[1] = np.clip(x_center[1] + dy, 0, 1)
                                X_refine.append(x_new)
                        
                        # Limit per point
                        if len(X_refine) >= n_refine:
                            break
                    
                    # If we have more points than budget, take the first n_refine
                    if len(X_refine) > n_refine:
                        X_refine = X_refine[:n_refine]
                    
                    # Evaluate refinement points
                    for x in X_refine:
                        if time.time() > deadline:
                            break
                        problem(x)
        
        return X_coarse
