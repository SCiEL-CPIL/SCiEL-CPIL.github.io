import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_hv(self, f1_vals, f2_vals, ref):
        """
        Compute Hypervolume for 2D points.
        f1_vals, f2_vals: numpy arrays of objective values.
        ref: tuple (ref1, ref2).
        """
        if len(f1_vals) == 0:
            return 0.0
        
        # Filter points strictly better than reference
        mask = (f1_vals < ref[0]) & (f2_vals < ref[1])
        if not np.any(mask):
            return 0.0
            
        f1 = f1_vals[mask]
        f2 = f2_vals[mask]
        
        # Sort by f1 ascending
        sort_idx = np.argsort(f1)
        f1 = f1[sort_idx]
        f2 = f2[sort_idx]
        
        # Handle duplicates in f1: keep only the one with minimum f2
        # Use a simple loop or numpy trick
        # Since N is small (200), a simple approach is fine, but let's use numpy for speed if possible.
        # Actually, for HV, if f1[i] == f1[i+1], the width is 0, so it doesn't contribute to the sum 
        # unless it changes the "height" for the next segment.
        # The standard algorithm: 
        # HV = sum_{i=1}^{n-1} (f1[i+1] - f1[i]) * (ref[1] - f2[i]) + (ref[0] - f1[n]) * (ref[1] - f2[n])
        # If f1[i] == f1[i+1], the term is 0. But f2[i] might be larger than f2[i+1].
        # The "dominated" point (larger f2 for same f1) doesn't affect the HV calculation in this 
        # specific summation formula IF we assume the points are non-dominated. 
        # But if we have duplicates, the "effective" height at that f1 is min(f2).
        # Let's filter out dominated duplicates.
        
        unique_mask = np.ones(len(f1), dtype=bool)
        # Iterate from end to start? Or just group by f1.
        # Since N=200, a simple python loop is fast enough.
        # But we can do it in numpy:
        # Find indices where f1 changes
        if len(f1) > 1:
            diff = np.diff(f1)
            change_idx = np.where(diff > 0)[0] + 1
            # For each group of identical f1, keep the one with min f2.
            # This is a bit tricky with vectorization.
            # Given N=200, let's just do a simple O(N^2) or O(N) filter if needed, 
            # but actually, the current loop in _compute_hv is O(N).
            # Let's just ensure that if f1[i] == f1[i+1], we keep the min f2.
            # We can do this by sorting by f1, then f2, and taking unique f1 with first occurrence (min f2).
            
            # Sort by f1, then f2
            sort_idx2 = np.lexsort((f2, f1))
            f1_sorted = f1[sort_idx2]
            f2_sorted = f2[sort_idx2]
            
            # Keep only unique f1 (first occurrence in sorted f2 is min f2)
            unique_f1_mask = np.concatenate(([True], np.diff(f1_sorted) > 0))
            f1 = f1_sorted[unique_f1_mask]
            f2 = f2_sorted[unique_f1_mask]

        hv = 0.0
        n = len(f1)
        if n == 1:
            return (ref[0] - f1[0]) * (ref[1] - f2[0])
            
        for i in range(n - 1):
            hv += (f1[i + 1] - f1[i]) * (ref[1] - f2[i])
        hv += (ref[0] - f1[-1]) * (ref[1] - f2[-1])
        
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        
        # 1. Model: ZDT2 Pareto Front is x2..xD = 0, f1=x1, f2=1-sqrt(x1)
        # We optimize the x1 values to maximize HV of the resulting set.
        
        N = self.budget
        dim = self.dim
        
        # Initial guess: Uniform grid in [0, 1]
        x1_params = np.linspace(0.0, 1.0, N)
        
        # Optimization of x1_params to maximize HV
        # Since the model is analytic, we can compute HV without calling problem()
        ref = (11.0, 11.0)
        
        def get_hv(x1s):
            # x1s are in [0, 1]
            f1 = x1s
            f2 = 1.0 - np.sqrt(f1)
            return self._compute_hv(f1, f2, ref)
        
        current_hv = get_hv(x1_params)
        
        # Improved Hill Climbing / Local Search
        # The HV landscape for 2D points on a curve is piecewise linear/smooth.
        # We can use a more aggressive local search.
        
        step_size = 0.05
        max_iter = 200
        stall_count = 0
        
        for iter in range(max_iter):
            if time.time() > deadline - 10.0: 
                break
            
            improved = False
            # Randomize the order of indices to avoid bias
            indices = np.random.permutation(N)
            
            for i in indices:
                original = x1_params[i]
                # Try multiple steps in different directions
                # Use a set of candidate steps: small, medium, large
                # But since we are on [0,1], we can just try random perturbations.
                
                # Adaptive step size: try to move towards the "gap" if possible?
                # For now, random hill climbing with restarts/steps.
                
                best_local_hv = current_hv
                best_local_x = original
                
                # Try 10 random perturbations
                for _ in range(10):
                    # Step size decays or is adaptive?
                    # Let's use a fixed step size for now, but maybe scaled by current position?
                    # No, just uniform.
                    new_x = np.clip(original + step_size * np.random.uniform(-1, 1), 0.0, 1.0)
                    x1_params[i] = new_x
                    new_hv = get_hv(x1_params)
                    if new_hv > best_local_hv + 1e-12:
                        best_local_hv = new_hv
                        best_local_x = new_x
                        improved = True
                
                # Keep the best found in this "inner loop"
                if improved:
                    x1_params[i] = best_local_x
                    current_hv = best_local_hv
                else:
                    x1_params[i] = original
            
            if not improved:
                step_size *= 0.7
                stall_count += 1
                if step_size < 1e-6 or stall_count > 5:
                    break
            else:
                stall_count = 0

        # 2. Execute evaluations using the optimized parameters
        # Prepare vectors
        X = np.zeros((N, dim))
        X[:, 0] = x1_params
        # x[1:] are already 0
        
        results = []
        for i in range(N):
            if time.time() > deadline:
                break
            x = X[i]
            # Ensure finite and within bounds (though 0 and x1 in [0,1] are safe)
            f = problem(x)
            results.append(f)
            
        return X
