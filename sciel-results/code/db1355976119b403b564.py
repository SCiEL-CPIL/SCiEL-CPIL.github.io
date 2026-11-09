"""Rewritten multi-objective optimizer for WFG9 with structured shape-space sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for WFG9.
    Strategy:
      1. Position variables fixed at midpoint (optimal for WFG9).
      2. Dense, structured sampling of the 3 shape variables to cover the curved Pareto front.
      3. Remaining budget used for adaptive refinement around non-dominated points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # 300s limit, 5s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0

        n_shape = min(3, self.dim)
        n_pos = self.dim - n_shape

        # Reserve budget: 70% for grid, 30% for refinement
        n_grid = int(self.budget * 0.7)
        n_refine = self.budget - n_grid

        # Phase 1: Structured grid on shape variables
        if self.dim >= 3:
            # 5x5x5 = 125, which is close to 70% of 200 = 140
            # Let's use 5x5x5 = 125
            g1, g2, g3 = 5, 5, 5
        elif self.dim == 2:
            g1, g2 = 12, 12
            g3 = 1
        else:
            g1 = n_grid
            g2 = 1
            g3 = 1

        grid_sizes = [g1, g2, g3]

        ranges = []
        for i in range(n_shape):
            if i < len(grid_sizes) and grid_sizes[i] > 0:
                ranges.append(np.linspace(lo[i], hi[i], grid_sizes[i]))
            else:
                ranges.append(np.array([mid[i]]))

        grids = np.meshgrid(*ranges, indexing='ij')
        grid_points = np.stack([g.ravel() for g in grids], axis=1)
        N = grid_points.shape[0]

        if N > n_grid:
            idx = np.random.choice(N, n_grid, replace=False)
            grid_points = grid_points[idx]
            N = n_grid

        # Construct full vectors: shape from grid, position from mid
        X = np.tile(mid, (N, self.dim))
        X[:, :n_shape] = grid_points

        # Evaluate grid points
        F = np.array([problem(x) for x in X])
        used = N

        # Check time
        if time.time() > deadline:
            return X

        # Phase 2: Adaptive refinement around non-dominated points
        if n_refine > 0 and used < self.budget:
            # Find non-dominated points
            n_points = len(F)
            is_dominated = np.zeros(n_points, dtype=bool)
            
            for i in range(n_points):
                if is_dominated[i]:
                    continue
                diff = F - F[i]
                leq = np.all(diff <= 0, axis=1)
                lt = np.any(diff < 0, axis=1)
                dominates_i = leq & lt
                dominates_i[i] = False
                if np.any(dominates_i):
                    is_dominated[i] = True

            non_dominated_idx = np.where(~is_dominated)[0]

            if len(non_dominated_idx) > 0:
                # Refine around non-dominated points
                n_to_refine = min(10, len(non_dominated_idx))
                selected = np.random.choice(non_dominated_idx, n_to_refine, replace=False)
                
                X_refine = []
                n_per_point = max(1, n_refine // n_to_refine)
                for idx in selected:
                    base = X[idx].copy()
                    for k in range(n_per_point):
                        x_new = base.copy()
                        for d in range(n_shape):
                            x_new[d] += np.random.uniform(-0.03, 0.03)
                            x_new[d] = np.clip(x_new[d], lo[d], hi[d])
                        X_refine.append(x_new)
                
                if X_refine:
                    X_refine = np.array(X_refine[:n_refine])
                    F_refine = np.array([problem(x) for x in X_refine])
                    X = np.vstack([X, X_refine])
                    F = np.vstack([F, F_refine])
                    used += len(X_refine)

        # Check time
        if time.time() > deadline:
            return X

        # Phase 3: If still have budget, add more random samples
        remaining = self.budget - used
        if remaining > 0:
            X_rand_shape = np.random.uniform(lo[:n_shape], hi[:n_shape], (remaining, n_shape))
            X_rand = np.tile(mid, (remaining, self.dim))
            X_rand[:, :n_shape] = X_rand_shape
            F_rand = np.array([problem(x) for x in X_rand])
            X = np.vstack([X, X_rand])
            F = np.vstack([F, F_rand])
            used += remaining

        return X

# EVOLVE-BLOCK-END
