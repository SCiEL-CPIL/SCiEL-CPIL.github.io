"""Rewritten program using Bayesian Optimization with GP surrogates and EHVI acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings

class Optimizer:
    """Bayesian Optimization multi-objective search using GP surrogates and EHVI."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _gp_predict(self, X_train, y_train, X_test, length_scale=0.5, noise=1e-6):
        """Simple GP prediction using RBF kernel. Returns mean and variance."""
        n = len(X_train)
        d = X_train.shape[1]
        
        # Compute kernel matrix
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                diff = X_train[i] - X_train[j]
                K[i, j] = np.exp(-0.5 * np.sum(diff ** 2) / (length_scale ** 2))
        K += noise * np.eye(n)
        
        # Kernel between train and test
        K_star = np.zeros((len(X_test), n))
        for i in range(len(X_test)):
            for j in range(n):
                diff = X_test[i] - X_train[j]
                K_star[i, j] = np.exp(-0.5 * np.sum(diff ** 2) / (length_scale ** 2))
        
        # Solve for alpha
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            v = np.linalg.solve(L, K_star.T)
            mu = K_star @ alpha
            var = 1.0 - np.sum(v ** 2, axis=1)
            var = np.maximum(var, 1e-12)
        except np.linalg.LinAlgError:
            # Fallback to least squares
            try:
                alpha = np.linalg.lstsq(K, y_train, rcond=None)[0]
                mu = K_star @ alpha
                var = np.ones(len(X_test)) * np.var(y_train)
            except:
                mu = np.full(len(X_test), np.mean(y_train))
                var = np.ones(len(X_test)) * np.var(y_train)
        
        return mu, var

    def _ehvi_score(self, X_candidates, F_train, ref_point):
        """
        Compute Expected Hypervolume Improvement for candidate points.
        Uses Monte Carlo sampling of the GP posterior for each objective.
        """
        n_cand = len(X_candidates)
        n_obj = len(ref_point)
        
        # For each candidate, we need to predict the distribution of objectives
        # We'll use a simplified EHVI: sample from GP, compute hypervolume contribution
        
        # First, compute the current hypervolume of the Pareto front
        # Filter points that are better than reference
        mask = np.all(F_train < ref_point, axis=1)
        F_valid = F_train[mask]
        
        if len(F_valid) == 0:
            current_hv = 0.0
        else:
            current_hv = self._hypervolume_3d(F_valid, ref_point)
        
        ehvi_scores = np.zeros(n_cand)
        
        # Number of MC samples
        n_samples = 50
        
        for i in range(n_cand):
            # Predict mean and variance for each objective at this candidate
            means = np.zeros(n_obj)
            stds = np.zeros(n_obj)
            
            for j in range(n_obj):
                mu_j, var_j = self._gp_predict(
                    self.X_train_all, 
                    self.F_train_all[:, j], 
                    X_candidates[i:i+1],
                    length_scale=0.5,
                    noise=1e-6
                )
                means[j] = mu_j[0]
                stds[j] = np.sqrt(var_j[0])
            
            # Sample from the posterior
            samples = means + stds * np.random.randn(n_samples, n_obj)
            
            # For each sample, compute the hypervolume contribution
            total_hv = 0.0
            for s in range(n_samples):
                f_sample = samples[s]
                if np.any(f_sample >= ref_point):
                    continue
                
                # Add this point to the front and compute new HV
                F_new = np.vstack([F_valid, f_sample.reshape(1, -1)])
                new_hv = self._hypervolume_3d(F_new, ref_point)
                total_hv += new_hv
            
            expected_hv = total_hv / n_samples
            ehvi_scores[i] = expected_hv - current_hv
        
        return ehvi_scores

    def _hypervolume_3d(self, points, ref_point):
        """Compute hypervolume for 3D points with given reference point."""
        if len(points) == 0:
            return 0.0
        
        # Filter points strictly better than reference
        mask = np.all(points < ref_point, axis=1)
        points = points[mask]
        
        if len(points) == 0:
            return 0.0
        
        # For 3D, we can use an O(n^2) algorithm
        # Sort by first objective
        points = points[np.argsort(points[:, 0])]
        
        hv = 0.0
        n = len(points)
        
        for i in range(n):
            # For point i, find the region it dominates
            # It contributes if it's not dominated by any previous point in the sorted order
            # with respect to the remaining objectives
            
            # Find the minimum values of objectives 1 and 2 among points that dominate point i
            # in objective 0 (i.e., points with f0 <= points[i][0])
            # Actually, since we sorted by f0 ascending, points 0..i have f0 <= points[i][0]
            
            # The volume contributed by point i is the volume of the box from points[i] to ref_point
            # minus the volume already covered by points 0..i-1 that are "to the left"
            
            # Simpler approach: for each point, compute the volume of the box it uniquely covers
            # Box for point i: [points[i][0], ref[0]] x [points[i][1], ref[1]] x [points[i][2], ref[2]]
            # But we need to subtract overlaps
            
            # A standard 3D HV algorithm:
            # Sort by f0 ascending
            # For each point i, find the max of (ref[1] - f1) and (ref[2] - f2) among points j < i
            # that have f1 <= points[i][1] or f2 <= points[i][2]... this gets complex.
            
            # Let's use a simpler but correct O(n^2) method:
            # For each point, compute the volume of the region it dominates that is not dominated by others.
            
            pass
        
        # Use a different approach: divide and conquer or use the fact that for small n,
        # we can compute it by inclusion-exclusion or by iterating.
        
        # Actually, let's use a well-known 3D HV computation:
        # Sort points by first objective (ascending)
        # Keep track of the "skyline" in the remaining 2D space
        
        # Sort by f0 ascending
        order = np.argsort(points[:, 0])
        sorted_points = points[order]
        
        # For 3D HV, we can use the following:
        # HV = sum over i of (ref[0] - f0[i]) * HV_2D of the 2D points (f1, f2) 
        #       that are not dominated by any point with f0 <= f0[i] in the 2D sense
        # But this is also complex.
        
        # Let's use a simple Monte Carlo or a direct computation for small n.
        # Since n is at most 200, and we call this many times, let's use an efficient method.
        
        # Alternative: use the formula for 3D hypervolume
        # Sort by f0 ascending
        # For each point i, the contribution is (ref[0] - f0[i]) * (area in 2D)
        # where the 2D area is the area of the region [f1[i], ref[1]] x [f2[i], ref[2]]
        # minus the area covered by previous points that have f1 <= f1[i] or f2 <= f2[i]
        
        # This is getting complex. Let's use a simpler approach:
        # For small n, we can compute HV by summing over all points the volume of the box
        # they define, minus overlaps. But for 3D, a cleaner way is:
        
        # Use the following algorithm:
        # 1. Sort points by f0 ascending
        # 2. For each point i, compute the 2D hypervolume of the set of points j (j <= i) 
        #    projected onto (f1, f2), considering only points with f0 <= f0[i]
        # 3. The contribution of point i is (ref[0] - f0[i]) * (HV_2D(i) - HV_2D(i-1))
        #    where HV_2D(i) is the 2D HV of points 0..i in the (f1, f2) space with ref (ref[1], ref[2])
        
        # Let's implement 2D HV first
        def hv_2d(pts_2d, ref_2d):
            """Compute 2D hypervolume for points with reference point."""
            if len(pts_2d) == 0:
                return 0.0
            # Sort by first coordinate ascending
            pts_2d = pts_2d[np.argsort(pts_2d[:, 0])]
            hv = 0.0
            max_y = ref_2d[1]
            for i in range(len(pts_2d)):
                x, y = pts_2d[i]
                if y < max_y:
                    hv += (ref_2d[0] - x) * (max_y - y)
                    max_y = y
            return hv
        
        # Now for 3D:
        sorted_pts = points[np.argsort(points[:, 0])]
        n_pts = len(sorted_pts)
        
        # We'll maintain a list of "active" points in 2D as we sweep through f0
        # For each point, we add it to the 2D set and compute the 2D HV
        # The contribution to 3D HV is (ref[0] - f0[i]) * (HV_2D_after - HV_2D_before)
        
        hv_3d = 0.0
        active_2d = []
        prev_hv_2d = 0.0
        
        for i in range(n_pts):
            f0, f1, f2 = sorted_pts[i]
            
            # Add this point to the 2D set
            active_2d.append([f1, f2])
            
            # Compute 2D HV of active_2d with reference (ref[1], ref[2])
            active_2d_arr = np.array(active_2d)
            # Filter points that are better than ref in 2D
            mask_2d = np.all(active_2d_arr < [ref_point[1], ref_point[2]], axis=1)
            active_2d_valid = active_2d_arr[mask_2d]
            
            curr_hv_2d = hv_2d(active_2d_valid, [ref_point[1], ref_point[2]])
            
            # The contribution of this point in the f0 dimension
            if i > 0:
                delta_hv_2d = curr_hv_2d - prev_hv_2d
            else:
                delta_hv_2d = curr_hv_2d
            
            # The volume in f0 dimension for this point
            # Points with f0 in [sorted_pts[i][0], sorted_pts[i+1][0]) or up to ref[0]
            if i < n_pts - 1:
                f0_width = sorted_pts[i+1][0] - sorted_pts[i][0]
            else:
                f0_width = ref_point[0] - sorted_pts[i][0]
            
            hv_3d += f0_width * delta_hv_2d
            
            prev_hv_2d = curr_hv_2d
        
        return hv_3d

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref_point = np.array([1705.0, 12.0, 0.3])
        
        # Initial random sampling
        n_init = min(20, self.budget)
        X_init = lo + (hi - lo) * rng.random((n_init, self.dim))
        F_init = np.array([problem(x) for x in X_init])
        
        X_all = X_init.copy()
        F_all = F_init.copy()
        used = n_init
        
        # Store training data for GP
        self.X_train_all = X_all.copy()
        self.F_train_all = F_all.copy()
        
        # Number of candidate points to evaluate for EHVI
        n_candidates = 100
        
        while used < self.budget:
            # Generate candidate points
            # Mix of random and perturbed points from current front
            n_random = n_candidates // 2
            n_perturb = n_candidates - n_random
            
            # Random candidates
            X_rand = lo + (hi - lo) * rng.random((n_random, self.dim))
            
            # Perturbed candidates from best points
            # Find non-dominated points
            dominated = np.zeros(len(F_all), dtype=bool)
            for i in range(len(F_all)):
                for j in range(len(F_all)):
                    if i != j and np.all(F_all[j] <= F_all[i]) and np.any(F_all[j] < F_all[i]):
                        dominated[i] = True
                        break
            nd_points = X_all[~dominated] if np.any(~dominated) else X_all
            
            X_perturb = []
            for _ in range(n_perturb):
                idx = rng.integers(len(nd_points))
                x_pert = nd_points[idx] + 0.2 * (hi - lo) * rng.standard_normal(self.dim)
                x_pert = np.clip(x_pert, lo, hi)
                X_perturb.append(x_pert)
            X_perturb = np.array(X_perturb)
            
            X_cand = np.vstack([X_rand, X_perturb])
            
            # Compute EHVI scores
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ehvi_scores = self._ehvi_score(X_cand, F_all, ref_point)
            
            # Select the best candidate
            best_idx = np.argmax(ehvi_scores)
            x_new = X_cand[best_idx]
            f_new = problem(x_new)
            used += 1
            
            # Update training data
            X_all = np.vstack([X_all, x_new.reshape(1, -1)])
            F_all = np.vstack([F_all, f_new.reshape(1, -1)])
            self.X_train_all = X_all.copy()
            self.F_train_all = F_all.copy()
        
        return X_all
# EVOLVE-BLOCK-END
