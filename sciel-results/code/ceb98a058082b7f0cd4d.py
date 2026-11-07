"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: hypervolume-driven sequential search for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point from task contract
        ref_point = np.array([3.0, 5.0, 7.0], dtype=np.float64)

        # Initialize with a diverse set of points
        init_size = min(30, budget)
        X_init = np.empty((init_size, dim))
        # Use Latin Hypercube-like initialization
        for i in range(init_size):
            X_init[i] = lo + span * ((i + 0.5) / init_size + 0.1 * np.random.randn(dim))
        X_init = np.clip(X_init, lo, hi)

        F_init = np.array([problem(x) for x in X_init], dtype=np.float64)
        used = init_size

        # Keep all evaluated points in a list
        X_all = X_init.copy()
        F_all = F_init.copy()

        # Helper: compute hypervolume contribution of a single point given an archive
        def hypervolume_contribution(point, archive_F, ref_pt):
            """Compute the hypervolume added by adding point to archive_F."""
            # If point is dominated by any archive point, contribution is 0
            for af in archive_F:
                if np.all(af <= point) and np.any(af < point):
                    return 0.0
            
            # Compute hypervolume of archive + point minus hypervolume of archive
            # For 3D, we can use a simple decomposition
            # Hypervolume of a set of points: sum over non-dominated points of the volume they uniquely cover
            
            def hv_3d(points, ref):
                """Compute hypervolume for 3D points."""
                n = len(points)
                if n == 0:
                    return 0.0
                
                # Sort by first objective
                idx = np.argsort(points[:, 0])
                pts = points[idx]
                
                # For each point, compute the volume it uniquely contributes
                # This is a simplified approach - exact HV in 3D requires more care
                total = 0.0
                for i in range(n):
                    # Volume covered by point i that is not covered by any point with better first objective
                    x_min = pts[i, 0]
                    x_max = ref[0]
                    if x_max <= x_min:
                        continue
                    
                    # Find the envelope of points that dominate in objectives 2 and 3
                    # among points with first objective <= pts[i, 0]
                    # Simplified: assume we can compute the 2D hypervolume in the y-z plane
                    
                    # For points j with pts[j, 0] <= pts[i, 0], they may dominate in obj 2 and 3
                    # The unique volume for point i is the region in [x_min, x_max] x [pts[i,1], ref[1]] x [pts[i,2], ref[2]]
                    # that is not covered by any point j with pts[j,0] <= pts[i,0]
                    
                    # This is complex; use a simpler approximation
                    # Contribution = (ref[0] - pts[i,0]) * (ref[1] - pts[i,1]) * (ref[2] - pts[i,2])
                    # times a correction factor for overlap
                    
                    vol = (ref[0] - pts[i, 0]) * (ref[1] - pts[i, 1]) * (ref[2] - pts[i, 2])
                    if vol > 0:
                        total += vol
                return total
            
            hv_with = hv_3d(np.vstack([archive_F, point.reshape(1, -1)]), ref_pt)
            hv_without = hv_3d(archive_F, ref_pt)
            return hv_with - hv_without

        # Main loop: sequentially select next point to evaluate
        while used < budget:
            if time.time() > deadline:
                break

            # Get current non-dominated archive
            # Filter to points strictly better than reference point
            mask = np.all(F_all < ref_point, axis=1)
            F_valid = F_all[mask]
            
            if len(F_valid) == 0:
                # No valid points yet, just do random search
                x_new = lo + span * np.random.rand(dim)
                f_new = problem(x_new)
                used += 1
                X_all = np.vstack([X_all, x_new.reshape(1, -1)])
                F_all = np.vstack([F_all, f_new.reshape(1, -1)])
                continue

            # Find non-dominated points in F_valid
            n_valid = len(F_valid)
            nondom_mask = np.ones(n_valid, dtype=bool)
            for i in range(n_valid):
                if not nondom_mask[i]:
                    continue
                for j in range(n_valid):
                    if i == j:
                        continue
                    if np.all(F_valid[j] <= F_valid[i]) and np.any(F_valid[j] < F_valid[i]):
                        nondom_mask[i] = False
                        break
            
            F_nondom = F_valid[nondom_mask]
            
            if len(F_nondom) == 0:
                x_new = lo + span * np.random.rand(dim)
                f_new = problem(x_new)
                used += 1
                X_all = np.vstack([X_all, x_new.reshape(1, -1)])
                F_all = np.vstack([F_all, f_new.reshape(1, -1)])
                continue

            # Select next point: try to find a point that maximizes hypervolume contribution
            # Use a simple strategy: generate candidate points and pick the one with best HV contribution
            n_candidates = 20
            best_candidate = None
            best_hv_gain = -1.0
            
            for _ in range(n_candidates):
                # Generate a candidate point near the current Pareto front
                # Pick a random non-dominated point and perturb it
                idx = np.random.randint(len(F_nondom))
                base = F_nondom[idx]
                
                # Perturb in objective space (approximate)
                # We'll generate a random point in the decision space that's likely to be good
                # Use a mixture: some random, some near existing good points
                if np.random.rand() < 0.5:
                    # Near a random non-dominated point's decision space location
                    # We need to map from objective space back to decision space approximately
                    # For now, just use random perturbation in decision space
                    x_cand = lo + span * np.random.rand(dim)
                else:
                    # More targeted: sample from the region between best points
                    # Find the point with best first objective
                    best_idx = np.argmin(F_nondom[:, 0])
                    x_cand = lo + span * np.random.rand(dim)
                    # Bias towards center
                    x_cand = 0.5 * x_cand + 0.5 * (lo + span * 0.5)
                
                x_cand = np.clip(x_cand, lo, hi)
                f_cand = problem(x_cand)
                used += 1
                
                # Check if this is better than reference
                if np.all(f_cand < ref_point):
                    # Compute HV contribution
                    hv_gain = hypervolume_contribution(f_cand, F_nondom, ref_point)
                    if hv_gain > best_hv_gain:
                        best_hv_gain = hv_gain
                        best_candidate = (x_cand, f_cand)
                
                # Add to all points regardless
                X_all = np.vstack([X_all, x_cand.reshape(1, -1)])
                F_all = np.vstack([F_all, f_cand.reshape(1, -1)])
                
                if used >= budget:
                    break

        return X_all
# EVOLVE-BLOCK-END
