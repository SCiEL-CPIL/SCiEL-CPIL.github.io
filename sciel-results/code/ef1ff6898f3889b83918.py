"""Bayesian Optimization with Gaussian Process surrogates and EHVI acquisition for RE34."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import itertools


class Optimizer:
    """Multi-objective Bayesian Optimizer using GP surrogates and EHVI acquisition."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_3d(self, points, ref):
        """Compute exact 3D hypervolume of a set of points w.r.t. reference point.
        Only points strictly better than ref in all objectives are considered.
        """
        if len(points) == 0:
            return 0.0
        # Filter points that are strictly better than ref in all objectives
        valid = np.all(points < ref, axis=1)
        if not np.any(valid):
            return 0.0
        pts = points[valid]
        
        # Sort by first objective
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        
        # Compute hypervolume in 3D using the standard algorithm
        # For each point, compute the volume of the box it uniquely dominates
        n = len(pts)
        hv = 0.0
        for i in range(n):
            # Point i dominates a box from pts[i] to ref
            # But we need to subtract overlaps with previously processed points
            # Standard approach: sort by f0, then for each point compute the 2D HV of the slice
            x0 = pts[i, 0]
            # The width in dimension 0 is determined by the next point or ref
            if i + 1 < n:
                x0_next = pts[i + 1, 0]
            else:
                x0_next = ref[0]
            width0 = x0_next - x0
            if width0 <= 0:
                continue
            
            # For the slice [x0, x0_next], the contributing points are those with f0 <= x0
            # Actually, let's use a cleaner 3D HV algorithm
            
        # Let's use a different, more robust 3D HV computation
        # Sort by f0 ascending
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        n = len(pts)
        
        # For each point i, the unique contribution is the 2D HV in the slice [f0_i, f0_{i+1}]
        # where f0_{n} = ref[0]
        hv = 0.0
        for i in range(n):
            f0_cur = pts[i, 0]
            if i + 1 < n:
                f0_next = pts[i + 1, 0]
            else:
                f0_next = ref[0]
            
            width0 = f0_next - f0_cur
            if width0 <= 0:
                continue
            
            # Points that dominate in dimensions 1 and 2 within this slice
            # These are points with index <= i (since sorted by f0, they have f0 <= f0_cur < f0_next)
            # But actually, for the slice [f0_cur, f0_next], the relevant points are those
            # with f0 in [f0_cur, f0_next), which is just point i (and possibly others with same f0)
            # Let's collect all points with f0 < f0_next
            mask = pts[:, 0] < f0_next
            sub_pts = pts[mask]
            
            # Compute 2D HV of sub_pts w.r.t. ref[1:] in the (f1, f2) plane
            # 2D HV: sort by f1, for each point compute (f2_i - ref[2]) * (f1_next - f1_i)
            if len(sub_pts) == 0:
                continue
            
            order1 = np.argsort(sub_pts[:, 1])
            sub_pts = sub_pts[order1]
            m = len(sub_pts)
            
            hv_2d = 0.0
            for j in range(m):
                f1_cur = sub_pts[j, 1]
                if j + 1 < m:
                    f1_next = sub_pts[j + 1, 1]
                else:
                    f1_next = ref[1]
                width1 = f1_next - f1_cur
                if width1 <= 0:
                    continue
                height2 = ref[2] - sub_pts[j, 2]
                if height2 > 0:
                    hv_2d += width1 * height2
            
            hv += width0 * hv_2d
        
        return hv

    def _compute_pareto_front(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated[j] = True
        return np.where(~dominated)[0]

    def _ehvi_acquisition(self, x, F, ref, mu, sigma):
        """Compute Expected Hypervolume Improvement at point x.
        Uses Monte Carlo approximation.
        """
        n_samples = 50
        rng = np.random.default_rng(42)
        
        # Sample from the GP posterior
        # mu: (n_obj,), sigma: (n_obj,) - diagonal covariance approximation
        # For simplicity, assume independent objectives
        samples = mu + sigma * rng.standard_normal((n_samples, self.n_obj))
        
        # Compute HV for current front
        pareto_idx = self._compute_pareto_front(F)
        if len(pareto_idx) == 0:
            F_pareto = F
        else:
            F_pareto = F[pareto_idx]
        
        hv_current = self._hypervolume_3d(F_pareto, ref)
        
        ehvi_sum = 0.0
        for s in range(n_samples):
            new_F = np.vstack([F, samples[s:s+1]])
            new_pareto = self._compute_pareto_front(new_F)
            if len(new_pareto) == 0:
                new_F_pareto = new_F
            else:
                new_F_pareto = new_F[new_pareto]
            hv_new = self._hypervolume_3d(new_F_pareto, ref)
            ehvi_sum += max(0.0, hv_new - hv_current)
        
        return ehvi_sum / n_samples

    def _fit_gp(self, X, F, dim):
        """Fit a simple GP surrogate for each objective.
        Returns mu and sigma functions.
        """
        n = len(X)
        if n == 0:
            return None
        
        # Use a simple radial basis kernel
        def kernel(X1, X2):
            # RBF kernel
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-0.5 * sq_dists)
        
        # Fit separate GPs for each objective
        mus = []
        sigmas = []
        for obj in range(self.n_obj):
            y = F[:, obj]
            K = kernel(X, X)
            # Add small jitter for numerical stability
            K += 1e-6 * np.eye(n)
            
            # Compute mean and variance
            # For a new point x, mu(x) = k(x, X) @ K^{-1} @ y
            # var(x) = k(x,x) - k(x,X) @ K^{-1} @ k(x,X)^T
            K_inv_y = np.linalg.solve(K, y)
            
            def mu_func(x, K_inv_y=K_inv_y, K=K, X=X):
                k = kernel(x.reshape(1, -1), X).flatten()
                return k @ K_inv_y
            
            def sigma_func(x, K=K, X=X):
                k = kernel(x.reshape(1, -1), X).flatten()
                var = 1.0 - k @ np.linalg.solve(K, k)
                return np.sqrt(max(var, 1e-10))
            
            mus.append(mu_func)
            sigmas.append(sigma_func)
        
        return mus, sigmas

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])
        
        # Initial random sampling
        n_init = min(20, self.budget)
        rng = np.random.default_rng(0)
        
        X = lo + (hi - lo) * rng.random((n_init, self.dim))
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Bayesian optimization loop
        while used < self.budget:
            # Fit GP surrogates
            gp = self._fit_gp(X, F, self.dim)
            if gp is None:
                # Fallback: random sampling
                x_new = lo + (hi - lo) * rng.random(self.dim)
                f_new = problem(x_new)
                X = np.vstack([X, x_new])
                F = np.vstack([F, f_new])
                used += 1
                continue
            
            mus, sigmas = gp
            
            # Find the best point using EHVI acquisition
            # Use multiple starting points for local optimization
            n_starts = 10
            best_x = None
            best_ehvi = -np.inf
            
            # Candidate starting points: current data points and random points
            starts = X.copy()
            rand_starts = lo + (hi - lo) * rng.random((n_starts, self.dim))
            starts = np.vstack([starts, rand_starts])
            
            for start in starts[:n_starts]:
                # Optimize the acquisition function locally
                def neg_ehvi(x):
                    # Compute mu and sigma at x
                    mu = np.array([mus[i](x) for i in range(self.n_obj)])
                    sigma = np.array([sigmas[i](x) for i in range(self.n_obj)])
                    # Clip sigma to be positive
                    sigma = np.maximum(sigma, 1e-8)
                    return -self._ehvi_acquisition(x, F, ref, mu, sigma)
                
                result = minimize(neg_ehvi, start, method='L-BFGS-B', bounds=list(zip(lo, hi)),
                                  options={'maxiter': 50, 'ftol': 1e-6})
                
                if result.success:
                    x_opt = np.clip(result.x, lo, hi)
                    mu = np.array([mus[i](x_opt) for i in range(self.n_obj)])
                    sigma = np.array([sigmas[i](x_opt) for i in range(self.n_obj)])
                    sigma = np.maximum(sigma, 1e-8)
                    ehvi = -result.fun
                    if ehvi > best_ehvi:
                        best_ehvi = ehvi
                        best_x = x_opt
            
            if best_x is None:
                # Fallback: random point
                best_x = lo + (hi - lo) * rng.random(self.dim)
            
            # Evaluate the new point
            f_new = problem(best_x)
            X = np.vstack([X, best_x])
            F = np.vstack([F, f_new])
            used += 1
            
            # Remove dominated points to keep the set manageable
            # Keep only non-dominated points plus some diversity
            pareto_idx = self._compute_pareto_front(F)
            if len(pareto_idx) < len(F) and len(pareto_idx) >= 5:
                # Keep Pareto front points and a few others for GP fitting
                # Actually, keep all points for GP fitting but use Pareto for HV
                pass
        
        return X
# EVOLVE-BLOCK-END
