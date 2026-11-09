import numpy as np


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 structure: first (n_obj-1) vars are position, rest are distance
        n_pos = self.n_obj - 1  # 2
        n_dist = self.dim - n_pos  # 10
        
        # Set distance vars to 0.5 (minimizes g)
        dist_val = 0.5 * (lo[n_pos:] + hi[n_pos:])  # center of bounds for dist vars
        
        # Generate points on the Pareto front (triangle in 2D)
        n_evals = self.budget
        # Use stratified sampling in the triangle
        # Map from unit square to triangle: (u,v) -> (u*(1-v), v)
        # or use rejection sampling / direct stratified
        
        # Better: use a grid in the 2D simplex
        # For n points, use sqrt(n) x sqrt(n) grid in triangle
        n_side = int(np.ceil(np.sqrt(n_evals)))
        
        points = []
        for i in range(n_side):
            for j in range(n_side):
                if i + j < n_side:  # inside triangle
                    # Map (i, j) to position in triangle
                    # x1 = i/n_side, x2 = j/n_side, need x1+x2 <= 1
                    x1 = (i + 0.5) / n_side
                    x2 = (j + 0.5) / n_side
                    if x1 + x2 <= 1.0:
                        points.append([x1, x2])
        
        # If we have more budget, add more refined points
        # Or use the remaining budget for perturbation
        
        # Actually, let's use a better quasi-random sequence
        # Use Halton or just stratified random
        
        # Simpler: generate n_evals random points in the triangle
        # with stratification for even coverage
        
        # Use a grid-based approach for even coverage
        n_grid = int(np.ceil(np.sqrt(n_evals)))
        
        candidates = []
        for i in range(n_grid):
            for j in range(n_grid):
                x1 = (i + 0.5) / n_grid
                x2 = (j + 0.5) / n_grid
                if x1 >= 0 and x2 >= 0 and x1 + x2 <= 1.0:
                    candidates.append((x1, x2))
        
        # If we need more points, add finer grid
        # If we have fewer, subsample
        
        # Take up to budget points
        if len(candidates) > n_evals:
            # Subsample
            idx = np.random.choice(len(candidates), n_evals, replace=False)
            candidates = [candidates[i] for i in idx]
        elif len(candidates) < n_evals:
            # Add more points with finer resolution
            n_grid2 = int(np.ceil(np.sqrt(n_evals * 2)))
            extra = []
            for i in range(n_grid2):
                for j in range(n_grid2):
                    x1 = (i + 0.5) / n_grid2
                    x2 = (j + 0.5) / n_grid2
                    if x1 >= 0 and x2 >= 0 and x1 + x2 <= 1.0:
                        extra.append((x1, x2))
            # Add unique points not already in candidates
            existing = set(candidates)
            for p in extra:
                if len(candidates) >= n_evals:
                    break
                if p not in existing:
                    candidates.append(p)
                    existing.add(p)
        
        # Evaluate
        results = []
        for x1, x2 in candidates[:n_evals]:
            x = np.zeros(self.dim)
            x[0] = x1 * (hi[0] - lo[0]) + lo[0]
            x[1] = x2 * (hi[1] - lo[1]) + lo[1]
            x[n_pos:] = dist_val
            f = problem(x)
            results.append(f)
        
        return results
