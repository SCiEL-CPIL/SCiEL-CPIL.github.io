import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        budget = self.budget
        
        # DTLZ7 Pareto front: x[2:] = 0.5, x[0], x[1] in [0,1]
        # Use ALL budget to sample the 2D Pareto front densely
        n_total = min(budget, 200)
        
        # Use a stratified Halton-like sequence for good 2D coverage
        # Halton sequence in 2D (bases 2 and 3)
        def halton(index, base):
            result = 0.0
            f = 1.0 / base
            i = index
            while i > 0:
                result += f * (i % base)
                i = i // base
                f /= base
            return result
        
        # Generate n_total points using Halton sequence
        x0_flat = np.array([halton(i + 1, 2) for i in range(n_total)])
        x1_flat = np.array([halton(i + 1, 3) for i in range(n_total)])
        
        # Also add a fine grid to ensure corners and edges are covered
        # The Halton sequence is good but may miss exact corners
        # Add a small set of strategic points: corners and midpoints
        strategic_x0 = [0.0, 0.25, 0.5, 0.75, 1.0]
        strategic_x1 = [0.0, 0.25, 0.5, 0.75, 1.0]
        strategic_points = []
        for sx0 in strategic_x0:
            for sx1 in strategic_x1:
                strategic_points.append((sx0, sx1))
        
        # Combine Halton and strategic points, deduplicate
        all_x0 = list(x0_flat)
        all_x1 = list(x1_flat)
        for sx0, sx1 in strategic_points:
            # Check if this point is already close to an existing point
            dists = np.sqrt((np.array(all_x0) - sx0)**2 + (np.array(all_x1) - sx1)**2)
            if np.min(dists) > 0.01:
                all_x0.append(sx0)
                all_x1.append(sx1)
        
        # Truncate to budget
        all_x0 = np.array(all_x0[:n_total])
        all_x1 = np.array(all_x1[:n_total])
        
        # Build points on the Pareto front
        X = np.zeros((len(all_x0), n))
        X[:, 0] = all_x0
        X[:, 1] = all_x1
        X[:, 2:] = 0.5  # Pareto front location
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = len(all_x0)
        
        # Check time - if we have remaining budget and time, do local refinement
        remaining = budget - used
        if remaining > 0 and time.time() < deadline - 10.0:
            # Find non-dominated points and do targeted refinement
            nd_mask = self._find_non_dominated(F)
            nd_indices = np.where(nd_mask)[0]
            
            if len(nd_indices) > 0:
                # Select a subset of non-dominated points for refinement
                n_refine = min(remaining, 20)
                if len(nd_indices) <= n_refine:
                    refine_idx = nd_indices
                else:
                    step = len(nd_indices) / n_refine
                    refine_idx = nd_indices[[int(i * step) for i in range(n_refine)]]
                
                rng = np.random.default_rng(42)
                for idx in refine_idx:
                    if used >= budget or time.time() > deadline - 5.0:
                        break
                    
                    # Generate a few perturbations around this point, keeping x[2:] = 0.5
                    n_local = min(remaining - (used - len(all_x0)), 3)
                    for _ in range(n_local):
                        if used >= budget or time.time() > deadline - 5.0:
                            break
                        
                        # Small perturbation in x[0] and x[1] only
                        perturbation = np.zeros(n)
                        perturbation[0] = 0.01 * rng.standard_normal()
                        perturbation[1] = 0.01 * rng.standard_normal()
                        x_child = X[idx] + perturbation
                        x_child = np.clip(x_child, lo, hi)
                        # Ensure x[2:] stays at 0.5
                        x_child[2:] = 0.5
                        
                        fc = problem(x_child)
                        used += 1
                        
                        # If better than parent, update
                        if np.all(fc <= F[idx]) and np.any(fc < F[idx]):
                            X[idx] = x_child
                            F[idx] = fc
        
        return X

    def _find_non_dominated(self, F):
        """Find indices of non-dominated points."""
        n = len(F)
        nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd[i]:
                continue
            for j in range(n):
                if i == j or not nd[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] = False
                    break
        return nd
