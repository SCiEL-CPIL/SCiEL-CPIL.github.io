"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for WFG9.
    Uses a structured approach:
    1. Dense grid on the first 3 (shape) variables with position variables at midpoints.
    2. Adaptive refinement using non-dominated sorting to fill gaps in the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _nondominated_sort(self, F):
        """Return indices of non-dominated points in F (minimization)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        if n == 1:
            return np.array([0], dtype=int)
        
        # For each point, check if any other point dominates it
        # Point i dominates j if F[i] <= F[j] in all objectives and < in at least one
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(i+1, n):
                if dominated[j]:
                    continue
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated[j] = True
                # Check if j dominates i
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        
        return np.where(~dominated)[0]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, 10s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        
        n_shape = min(3, self.dim)
        
        # Phase 1: Grid on shape variables
        # 6x6x4 = 144 points for grid, leaving 56 for refinement
        if self.dim >= 3:
            g1, g2, g3 = 6, 6, 4
        elif self.dim == 2:
            g1, g2 = 12, 12
            g3 = 1
        else:
            g1 = 144
            g2 = 1
            g3 = 1
            
        grid_sizes = [g1, g2, g3]
        
        # Generate grid points for shape variables
        ranges = []
        for i in range(n_shape):
            if i < len(grid_sizes) and grid_sizes[i] > 0:
                ranges.append(np.linspace(lo[i], hi[i], grid_sizes[i]))
            else:
                ranges.append(np.array([mid[i]]))
        
        grids = np.meshgrid(*ranges, indexing='ij')
        grid_points = np.stack([g.ravel() for g in grids], axis=1)
        N = grid_points.shape[0]
        
        # Construct full vectors: shape from grid, position from mid
        X = np.tile(mid, (N, 1))
        X[:, :n_shape] = grid_points
        
        # Evaluate grid points
        F = np.array([problem(x) for x in X])
        used = N
        
        # Check time
        if time.time() > deadline:
            return X
        
        # Phase 2: Adaptive refinement via interpolation
        remaining = self.budget - used
        if remaining > 0:
            # Get current non-dominated points
            nd_idx = self._nondominated_sort(F)
            
            if len(nd_idx) > 1:
                # Sort non-dominated points by first objective to find neighbors
                nd_F = F[nd_idx]
                sorted_order = np.argsort(nd_F[:, 0])
                sorted_nd_idx = nd_idx[sorted_order]
                
                # Generate candidates by interpolating between adjacent non-dominated points
                candidates = []
                # We want to use about 75% of remaining budget for interpolation
                # and 25% for random polish
                n_interp = int(remaining * 0.75)
                
                # Create a list of interpolation tasks
                # For each pair of adjacent ND points, try t=0.5
                # If we run out of budget, we stop.
                # We can also try t=0.25, 0.75 if budget allows.
                # Let's just do t=0.5 for all pairs first, then fill rest with random.
                
                for i in range(len(sorted_nd_idx) - 1):
                    if len(candidates) >= n_interp:
                        break
                    idx1 = sorted_nd_idx[i]
                    idx2 = sorted_nd_idx[i+1]
                    
                    # Interpolate in decision space
                    x1 = X[idx1]
                    x2 = X[idx2]
                    
                    # Try midpoint
                    x_mid = 0.5 * x1 + 0.5 * x2
                    candidates.append(x_mid)
                    
                    # If we have more budget, try quartiles
                    if len(candidates) < n_interp:
                        x_q1 = 0.25 * x1 + 0.75 * x2
                        candidates.append(x_q1)
                    if len(candidates) < n_interp:
                        x_q3 = 0.75 * x1 + 0.25 * x2
                        candidates.append(x_q3)

                # Evaluate interpolation candidates
                if candidates:
                    X_cand = np.array(candidates[:n_interp])
                    # Ensure they are within bounds (they should be if x1, x2 are)
                    F_cand = np.array([problem(x) for x in X_cand])
                    used += len(X_cand)
                    
                    X = np.vstack([X, X_cand])
                    F = np.vstack([F, F_cand])
                    
                    # Check time
                    if time.time() > deadline:
                        return X

            # Phase 3: Final polish - random perturbations around non-dominated points
            remaining2 = self.budget - used
            if remaining2 > 0 and time.time() < deadline:
                nd_idx = self._nondominated_sort(F)
                if len(nd_idx) > 0:
                    X_polish = np.zeros((remaining2, self.dim))
                    for k in range(remaining2):
                        parent_idx = nd_idx[np.random.randint(len(nd_idx))]
                        X_polish[k] = X[parent_idx].copy()
                        
                        # Small Gaussian noise on shape variables
                        for d in range(n_shape):
                            scale = (hi[d] - lo[d]) / max(grid_sizes[d], 2)
                            noise = np.random.normal(0, scale * 0.2)
                            X_polish[k, d] += noise
                            X_polish[k, d] = np.clip(X_polish[k, d], lo[d], hi[d])
                        
                        # Keep position variables at mid
                        for d in range(n_shape, self.dim):
                            X_polish[k, d] = mid[d]
                    
                    F_polish = np.array([problem(x) for x in X_polish])
                    used += remaining2
                    
                    X = np.vstack([X, X_polish])
                    F = np.vstack([F, F_polish])
        
        return X
# EVOLVE-BLOCK-END
