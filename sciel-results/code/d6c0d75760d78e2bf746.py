"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Surrogate-assisted multi-objective optimizer using GP + EHVI-style acquisition."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Reference point for hypervolume
        ref = np.array([1705.0, 12.0, 0.3])
        
        # Initialize with Latin Hypercube Sampling for better coverage
        n_init = min(30, budget // 4)
        rng = np.random.default_rng(0)
        
        # Latin Hypercube
        X_init = np.zeros((n_init, dim))
        for d in range(dim):
            perm = rng.permutation(n_init)
            X_init[:, d] = (perm + rng.random(n_init)) / n_init
        X_init = lo + (hi - lo) * X_init
        
        # Evaluate initial population
        X = X_init.copy()
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Track all evaluated points for hypervolume
        all_X = X.copy()
        all_F = F.copy()
        
        # GP surrogate components
        def kernel(X1, X2, length_scales):
            """Matern 5/2 kernel"""
            # Standardize
            D = np.sqrt(np.sum((X1[:, None, :] - X2[None, :, :])**2 * (1.0 / length_scales**2), axis=2))
            sqrt5D = np.sqrt(5.0) * D
            return (1.0 + sqrt5D + D**2 / 3.0) * np.exp(-sqrt5D)
        
        def log_likelihood(theta, X, F):
            """Compute log marginal likelihood for GP"""
            n = len(X)
            length_scales = np.exp(theta[:dim])
            sigma_f = np.exp(theta[dim])
            sigma_n = np.exp(theta[dim + 1])
            
            K = kernel(X, X, length_scales)
            K[np.diag_indices_from(K)] += sigma_n**2
            y = F
            
            # Add jitter for numerical stability
            K += 1e-6 * np.eye(n)
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                mll = -0.5 * y @ alpha - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
                return mll
            except np.linalg.LinAlgError:
                return -1e10
        
        def predict(X_new, X, F, theta):
            """Predict mean and variance at X_new given training data"""
            length_scales = np.exp(theta[:dim])
            sigma_f = np.exp(theta[dim])
            sigma_n = np.exp(theta[dim + 1])
            
            K = kernel(X, X, length_scales)
            K[np.diag_indices_from(K)] += sigma_n**2
            K += 1e-6 * np.eye(len(X))
            
            Ks = kernel(X_new, X, length_scales)
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, F))
                v = np.linalg.solve(L, Ks.T)
                mean = Ks @ alpha
                var = sigma_f**2 - np.sum(v**2, axis=1)
                var = np.maximum(var, 1e-10)
                return mean, var
            except np.linalg.LinAlgError:
                # Fallback: return prior
                return np.zeros(len(X_new)), np.ones(len(X_new)) * sigma_f**2
        
        def hypervolume_2d(points, ref):
            """Compute hypervolume in 2D for a set of points"""
            # Filter points dominated by ref
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            prev_y = ref[1]
            for x, y in pts:
                if y < prev_y:
                    hv += (ref[0] - x) * (prev_y - y)
                    prev_y = y
            return hv
        
        def ehvi_acquisition(x, X, F, theta, ref):
            """Expected Hypervolume Improvement acquisition function"""
            # Simplified EHVI: use expected improvement on a weighted sum as proxy
            # This is a heuristic approximation
            try:
                mean, var = predict(x.reshape(1, -1), X, F, theta)
                mean = mean[0]
                var = var[0]
                
                # Use a simple EI-like metric: expected reduction in weighted sum
                # Weight by inverse of reference point scale
                weights = 1.0 / ref
                current_best = np.min(np.sum(F * weights, axis=1))
                expected = np.sum(mean * weights)
                sigma = np.sqrt(np.sum(var * weights**2))
                
                if sigma < 1e-10:
                    return -(current_best - expected)
                
                z = (current_best - expected) / sigma
                from scipy.stats import norm
                ei = (current_best - expected) * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei  # Negate since we want to maximize
            except Exception:
                return 0.0
        
        # Main optimization loop
        while used < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Fit GP to current data (use subset if too large for speed)
            max_gp_points = min(len(X), 50)
            if len(X) > max_gp_points:
                # Use non-dominated points plus random subset
                dominated = np.array([np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F])
                nondom_idx = np.where(~dominated)[0]
                if len(nondom_idx) > max_gp_points:
                    idx = np.random.default_rng(used).choice(nondom_idx, max_gp_points, replace=False)
                else:
                    remaining = max_gp_points - len(nondom_idx)
                    other_idx = np.where(dominated)[0]
                    if len(other_idx) >= remaining:
                        extra = np.random.default_rng(used).choice(other_idx, remaining, replace=False)
                        idx = np.concatenate([nondom_idx, extra])
                    else:
                        idx = np.concatenate([nondom_idx, other_idx])
            else:
                idx = np.arange(len(X))
            
            X_gp = X[idx]
            F_gp = F[idx]
            
            # Optimize GP hyperparameters
            initial_theta = np.zeros(dim + 2)
            initial_theta[:dim] = np.log(np.ones(dim) * (hi - lo) / 2)
            initial_theta[dim] = np.log(np.std(F_gp, axis=0) + 1e-6).mean()
            initial_theta[dim + 1] = 0.0
            
            best_theta = initial_theta
            best_ll = log_likelihood(initial_theta, X_gp, F_gp)
            
            # Simple local optimization of hyperparameters
            def neg_ll(theta):
                return -log_likelihood(theta, X_gp, F_gp)
            
            try:
                res = minimize(neg_ll, initial_theta, method='Nelder-Mead', 
                              options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-3})
                if res.fun < best_ll:
                    best_theta = res.x
                    best_ll = -res.fun
            except Exception:
                pass
            
            # Generate candidate points using multi-start local optimization
            n_candidates = 10
            candidates = []
            
            for c in range(n_candidates):
                if time.time() > deadline - 5:
                    break
                
                # Start from random point or existing point
                if c == 0:
                    x0 = lo + (hi - lo) * np.random.default_rng(used + c).random(dim)
                else:
                    # Perturb existing points
                    base_idx = np.random.default_rng(used + c).integers(len(X))
                    x0 = X[base_idx] + 0.05 * (hi - lo) * np.random.default_rng(used + c).standard_normal(dim)
                    x0 = np.clip(x0, lo, hi)
                
                try:
                    res_c = minimize(ehvi_acquisition, x0, args=(X, F, best_theta, ref),
                                   method='Nelder-Mead', bounds=list(zip(lo, hi)),
                                   options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
                    candidates.append(np.clip(res_c.x, lo, hi))
                except Exception:
                    candidates.append(x0)
            
            # Select best candidate
            if len(candidates) > 0:
                # Evaluate acquisition function for all candidates
                acq_vals = np.array([ehvi_acquisition(c, X, F, best_theta, ref) for c in candidates])
                best_candidate = candidates[np.argmin(acq_vals)]
                
                # Evaluate the selected point
                fc = problem(best_candidate)
                used += 1
                
                # Add to population
                X = np.vstack([X, best_candidate])
                F = np.vstack([F, fc])
                all_X = np.vstack([all_X, best_candidate])
                all_F = np.vstack([all_F, fc])
                
                # Trim population to keep GP tractable
                if len(X) > 60:
                    # Keep non-dominated points and remove worst dominated
                    dominated = np.array([np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F])
                    if np.sum(dominated) > 10:
                        # Remove worst dominated points
                        dom_idx = np.where(dominated)[0]
                        # Sort by sum of objectives (worse first)
                        dom_scores = np.sum(F[dom_idx], axis=1)
                        worst_idx = dom_idx[np.argsort(dom_scores)[-10:]]
                        keep_mask = np.ones(len(X), dtype=bool)
                        keep_mask[worst_idx] = False
                        X = X[keep_mask]
                        F = F[keep_mask]
        
        return all_X
# EVOLVE-BLOCK-END
