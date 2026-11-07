"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """Model-Based Evolutionary Algorithm with GP surrogate and EHVI acquisition."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        # Initial design: LHS for space-filling
        n_init = min(20, self.budget)
        X = self._lhs_design(n_init, self.dim, lo, hi, rng)
        F = np.array([problem(x) for x in X])
        used = n_init

        # Reference point for hypervolume calculation
        ref_point = np.array([1.1, 1.3, 1.1])

        # Store all evaluated points
        all_X = X.copy()
        all_F = F.copy()

        # Build initial GP surrogates
        # Use a simple RBF kernel GP for each objective
        def gp_predict(X_train, y_train, X_pred, length_scale=0.3, sigma_f=0.1, sigma_n=1e-6):
            """Simple GP prediction with RBF kernel."""
            n = len(X_train)
            # Compute kernel matrix
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    diff = X_train[i] - X_train[j]
                    K[i, j] = sigma_f ** 2 * np.exp(-0.5 * np.sum(diff ** 2) / (length_scale ** 2))
            K += sigma_n ** 2 * np.eye(n)
            K_inv = np.linalg.inv(K)
            # Predict mean and variance
            mu = np.zeros(len(X_pred))
            var = np.zeros(len(X_pred))
            for i in range(len(X_pred)):
                k_star = np.zeros(n)
                for j in range(n):
                    diff = X_pred[i] - X_train[j]
                    k_star[j] = sigma_f ** 2 * np.exp(-0.5 * np.sum(diff ** 2) / (length_scale ** 2))
                mu[i] = k_star @ K_inv @ y_train
                var[i] = sigma_f ** 2 - k_star @ K_inv @ k_star
                var[i] = max(var[i], 1e-8)
            return mu, var

        def hypervolume_3d(points, ref):
            """Compute 3D hypervolume using inclusion-exclusion."""
            # Filter points dominated by ref
            valid = np.all(points < ref, axis=1)
            pts = points[valid]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            # Use sweep line for 3D
            # For simplicity, use a grid-based approximation
            n_grid = 20
            h = np.linspace(pts[:, 0].min(), ref[0], n_grid)
            hv = 0.0
            for i in range(n_grid - 1):
                dh = h[i + 1] - h[i]
                # Points with f0 <= h[i+1]
                mask = pts[:, 0] <= h[i + 1]
                sub_pts = pts[mask]
                if len(sub_pts) == 0:
                    continue
                # 2D hypervolume in remaining 2 dims
                hv += dh * self._hv_2d(sub_pts[:, 1:], ref[1:])
            return hv

        def _hv_2d(self, pts, ref):
            """2D hypervolume."""
            if len(pts) == 0:
                return 0.0
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            hv = 0.0
            max_y = ref[1]
            for i in range(len(pts)):
                if pts[i, 0] >= ref[0]:
                    break
                if pts[i, 1] < max_y:
                    hv += (min(pts[i, 0], ref[0]) - (pts[i - 1, 0] if i > 0 else 0)) * (max_y - pts[i, 1])
                    max_y = pts[i, 1]
            # Simplified: just sum boxes
            hv = 0.0
            sorted_pts = pts[np.argsort(pts[:, 0])]
            prev_x = 0.0
            for p in sorted_pts:
                if p[0] >= ref[0]:
                    break
                w = min(p[0], ref[0]) - prev_x
                h = ref[1] - p[1]
                if h > 0:
                    hv += w * h
                prev_x = p[0]
            return hv

        def compute_ehvi(X_pred, F_pred_mean, F_pred_var, ref, all_F):
            """Compute EHVI for a set of candidate points."""
            # Get current Pareto front
            pareto_mask = self._is_pareto(all_F)
            pareto_F = all_F[pareto_mask]
            current_hv = self._hypervolume(pareto_F, ref)
            
            ehvi = np.zeros(len(X_pred))
            for i in range(len(X_pred)):
                # Add predicted point to front
                new_F = np.vstack([all_F, F_pred_mean[i:i+1]])
                new_pareto = self._is_pareto(new_F)
                new_hv = self._hypervolume(new_F[new_pareto], ref)
                # Approximate EHVI using normal distribution
                delta_hv = new_hv - current_hv
                if delta_hv > 0:
                    z = delta_hv / np.sqrt(np.sum(F_pred_var[i]))
                    ehvi[i] = delta_hv * norm.cdf(z) + np.sqrt(np.sum(F_pred_var[i])) * norm.pdf(z)
            return ehvi

        def _is_pareto(self, F):
            """Return boolean mask for Pareto-optimal points."""
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i != j and mask[j]:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            mask[i] = False
                            break
            return mask

        def _hypervolume(self, points, ref):
            """Compute hypervolume using simple box decomposition for 3D."""
            if len(points) == 0:
                return 0.0
            valid = np.all(points < ref, axis=1)
            pts = points[valid]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            hv = 0.0
            # Use a simple grid-based approach
            n_grid = 15
            h0 = np.linspace(pts[:, 0].min(), ref[0], n_grid)
            for i in range(n_grid - 1):
                mask = pts[:, 0] <= h0[i + 1]
                sub = pts[mask]
                if len(sub) == 0:
                    continue
                hv += (h0[i + 1] - h0[i]) * self._hv_2d(sub[:, 1:], ref[1:])
            return hv

        # Main optimization loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Build GP models for each objective
            F_mean = np.zeros((self.budget - used + 1, self.n_obj))
            F_var = np.zeros((self.budget - used + 1, self.n_obj))
            
            # Generate candidate points
            n_candidates = 50
            X_cand = lo + (hi - lo) * rng.random((n_candidates, self.dim))
            
            # Also add some points near existing Pareto front
            pareto_mask = self._is_pareto(all_F)
            if np.any(pareto_mask):
                pareto_X = all_X[pareto_mask]
                n_near = min(20, len(pareto_X))
                idx_near = rng.choice(len(pareto_X), size=n_near, replace=False)
                X_near = pareto_X[idx_near] + 0.1 * (hi - lo) * rng.standard_normal((n_near, self.dim))
                X_near = np.clip(X_near, lo, hi)
                X_cand = np.vstack([X_cand, X_near])
            
            # Predict with GP
            for obj in range(self.n_obj):
                mu, var = gp_predict(all_X, all_F[:, obj], X_cand)
                F_mean[:, obj] = mu
                F_var[:, obj] = var
            
            # Compute EHVI
            ehvi = compute_ehvi(X_cand, F_mean, F_var, ref_point, all_F)
            
            # Select best candidate
            best_idx = np.argmax(ehvi)
            x_new = X_cand[best_idx]
            f_new = problem(x_new)
            used += 1
            
            # Add to archive
            all_X = np.vstack([all_X, x_new])
            all_F = np.vstack([all_F, f_new])
            
            # Prune archive to keep only Pareto-optimal points (limit size)
            if len(all_F) > 100:
                pareto_mask = self._is_pareto(all_F)
                if np.sum(pareto_mask) > 50:
                    # Keep all Pareto points
                    all_X = all_X[pareto_mask]
                    all_F = all_F[pareto_mask]
                else:
                    # Keep Pareto points plus some diverse non-Pareto points
                    pareto_idx = np.where(pareto_mask)[0]
                    non_pareto_idx = np.where(~pareto_mask)[0]
                    if len(non_pareto_idx) > 0:
                        # Select diverse non-Pareto points
                        n_keep = min(10, len(non_pareto_idx))
                        # Simple diversity: max min distance
                        selected = [pareto_idx[0]]
                        remaining = list(non_pareto_idx)
                        for _ in range(n_keep):
                            if not remaining:
                                break
                            best_dist = -1
                            best_j = -1
                            for j in remaining:
                                min_dist = min(np.linalg.norm(all_X[j] - all_X[k]) for k in selected)
                                if min_dist > best_dist:
                                    best_dist = min_dist
                                    best_j = j
                            selected.append(best_j)
                            remaining.remove(best_j)
                        keep_idx = np.array(selected + list(pareto_idx))
                        all_X = all_X[keep_idx]
                        all_F = all_F[keep_idx]
        
        return all_X

    def _lhs_design(self, n, dim, lo, hi, rng):
        """Latin Hypercube Sampling."""
        X = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            X[:, d] = (perm + rng.random(n)) / n
        X = lo + (hi - lo) * X
        return X
# EVOLVE-BLOCK-END
