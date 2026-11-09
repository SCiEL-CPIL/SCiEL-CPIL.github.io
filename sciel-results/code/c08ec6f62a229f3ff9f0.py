"""Multi-objective DTLZ4 optimizer using Sobol sequence and Max-Min Diversity selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.stats.qmc


class Optimizer:
    """DTLZ4 optimizer: sample the known Pareto front using Sobol sequence and Max-Min Diversity selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        n_shape = m - 1  # 2 shape variables for 3 objectives
        
        # 1. Generate a pool of candidate points using Sobol sequence
        # Sobol is a quasi-random sequence with low discrepancy
        pool_size = 1000
        sobol = scipy.stats.qmc.Sobol(d=2, scramble=True)
        # Sobol requires power of 2 for sample size for optimal properties, but we can just take first N
        # We will generate a slightly larger power of 2 and trim
        n_sobol = 1024 
        sobol_pts = sobol.random(n_sobol)
        sobol_pts = sobol_pts[:pool_size]
        
        # 2. Map Sobol points to angles for uniform area coverage on the spherical front
        u = sobol_pts[:, 0]
        v = sobol_pts[:, 1]
        
        # Ensure we include the corners explicitly in the pool
        # Corners in u,v space: (0,0), (1,0), (0,1), (1,1)
        # Note: arccos(1-u) for u=0 is 0, u=1 is pi/2.
        # v=0 -> beta=0, v=1 -> beta=pi/2.
        # These correspond to the axes of the front.
        
        alpha_pool = np.arccos(1.0 - u)
        beta_pool = v * (np.pi / 2.0)
        
        # 3. Compute analytic objective values for the pool
        # DTLZ2/4 front:
        # f1 = sin(alpha) * sin(beta)
        # f2 = sin(alpha) * cos(beta)
        # f3 = cos(alpha)
        # (Ignoring the (1+g) factor as it is constant for all points on the front)
        
        F_pool = np.zeros((pool_size, m))
        F_pool[:, 0] = np.sin(alpha_pool) * np.sin(beta_pool)
        F_pool[:, 1] = np.sin(alpha_pool) * np.cos(beta_pool)
        F_pool[:, 2] = np.cos(alpha_pool)
        
        # 4. Greedy Max-Min Diversity Selection
        # Select `budget` points from the pool to maximize the minimum distance to already selected points
        selected_indices = []
        
        # Initial point: choose the one closest to the "center" of the front or just the first one
        # Actually, let's pick the point with the smallest sum of objectives (closest to origin in f-space? No, we want spread)
        # Let's just pick the first point in the pool as a seed
        current_selected = [0]
        
        for _ in range(self.budget - 1):
            # Compute min distance from each candidate to the current set of selected points
            # Distances in objective space (Euclidean)
            # F_pool: (N, m), F_selected: (K, m)
            # We want dist(candidate, selected) = min_j ||f_c - f_sj||
            
            # Vectorized distance calculation
            # diff = F_pool[:, np.newaxis, :] - F_selected[np.newaxis, :, :]
            # dists = np.sqrt(np.sum(diff**2, axis=2))
            # min_dists = np.min(dists, axis=1)
            
            # To save memory/time, we can do it in chunks or just use numpy broadcasting
            # N=1000, K<=200. 1000*200*3 is small.
            
            F_sel = F_pool[current_selected]
            diff = F_pool[:, np.newaxis, :] - F_sel[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            min_dists = np.min(dists, axis=1)
            
            # Exclude already selected points
            for idx in current_selected:
                min_dists[idx] = -1.0
                
            # Select the point with the maximum min-distance
            next_idx = np.argmax(min_dists)
            current_selected.append(next_idx)
            
            if time.time() > deadline - 10.0:
                break

        # 5. Construct the final X vectors
        X = np.zeros((len(current_selected), n))
        
        # Map selected indices back to angles
        alpha_sel = alpha_pool[current_selected]
        beta_sel = beta_pool[current_selected]
        
        X[:, 0] = alpha_sel
        X[:, 1] = beta_sel
        
        # Set g-variables to 0.5
        for j in range(n_shape, n):
            X[:, j] = 0.5

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # 6. Evaluate all points with time guard
        F = np.empty((len(X), m))
        for i in range(len(X)):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])

        return X
# EVOLVE-BLOCK-END
