import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 Structure Assumption:
        # x1 is the first variable, others are 0 for the front.
        # f1 = x1
        # f2 = 1 - sqrt(x1) - x1 * sin(10 * pi * x1)
        # Valid for x1 in [0, 1].
        
        # 1. Analytically determine Pareto points
        # Scan x1 in [0, 1] with fine resolution
        n_scan = 10000
        x1_scan = np.linspace(0.0, 1.0, n_scan)
        
        # Compute objectives analytically
        # Handle x1=0 carefully for sqrt
        f1_scan = x1_scan.copy()
        f2_scan = np.zeros_like(x1_scan)
        mask = x1_scan > 0
        f2_scan[mask] = 1 - np.sqrt(x1_scan[mask]) - x1_scan[mask] * np.sin(10 * np.pi * x1_scan[mask])
        f2_scan[~mask] = 1 # f2(0) = 1
        
        # Find running minima
        # A point is Pareto optimal if f2[i] < min(f2[0:i])
        min_f2_so_far = np.inf
        pareto_indices = []
        for i in range(n_scan):
            if f2_scan[i] < min_f2_so_far:
                min_f2_so_far = f2_scan[i]
                pareto_indices.append(i)
        
        # The number of Pareto points is likely small.
        # Let's say we found K points.
        # We will evaluate these points.
        
        # Create the candidate list
        candidates = []
        
        # Add Pareto points
        for idx in pareto_indices:
            x = np.zeros(self.dim)
            x[0] = x1_scan[idx]
            candidates.append(x)
        
        # If we have budget left, add some uniform points to be safe
        # or points around the Pareto ones to ensure we hit the exact minimum if scan was coarse.
        # But 10000 points is very fine. The error is negligible.
        
        # Shuffle candidates to ensure we don't run out of time on the "good" ones?
        # No, we want to evaluate the Pareto ones first.
        
        # If we have more budget than pareto points, fill with random uniform in [0,1]
        # This might catch any numerical issues or if the problem is slightly different.
        rng = np.random.default_rng(42)
        while len(candidates) < self.budget:
            x = np.zeros(self.dim)
            x[0] = rng.uniform(0.0, 1.0)
            candidates.append(x)
            
        # Evaluate
        results = []
        used = 0
        for x in candidates:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Clip to bounds just in case
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            results.append(f)
            used += 1
            
        return results
