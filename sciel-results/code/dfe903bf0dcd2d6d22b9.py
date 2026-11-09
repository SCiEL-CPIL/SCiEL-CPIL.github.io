"""Optimizer for BBOB-style f13 (Sharp ridge) with ridge-aware search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Sobol + ridge-aware local search for BBOB f13 (Sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 275.0  # safety margin before 300s kill
        n = self.dim
        budget = self.budget

        def elapsed():
            return time.time() - t0

        calls = 0
        best_f = np.inf
        best_x = np.zeros(n)

        # Store all evaluations for ridge detection
        eval_points = []
        eval_values = []

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or elapsed() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                return best_f
            if not np.isfinite(f):
                return best_f
            calls += 1
            eval_points.append(x.copy())
            eval_values.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Sobol broad exploration
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=n, scramble=True, seed=42)
            sobol.random(1)  # skip origin
            n_init = min(60, budget // 5)
            pts = sobol.random(n_init) * 10.0 - 5.0
        except Exception:
            rng_init = np.random.default_rng(42)
            n_init = min(60, budget // 5)
            pts = rng_init.random((n_init, n)) * 10.0 - 5.0

        for i in range(n_init):
            if calls >= budget or elapsed() > deadline:
                break
            safe_call(pts[i])

        # Phase 2: Ridge direction estimation using top-k points
        # f13 has a sharp ridge structure; we estimate the ridge direction
        # by finding the principal component of the best points weighted by fitness

        def estimate_ridge_direction():
            """Estimate ridge direction from top-k evaluated points."""
            if len(eval_points) < 10:
                return None
            pts_arr = np.array(eval_points)
            vals_arr = np.array(eval_values)
            # Take top-k points (lowest function values)
            k = min(20, len(pts_arr))
            idx = np.argsort(vals_arr)[:k]
            top_pts = pts_arr[idx]
            top_vals = vals_arr[idx]
            
            # Weight by inverse fitness (better points get more weight)
            min_val = np.min(top_vals)
            max_val = np.max(top_vals)
            if max_val - min_val < 1e-10:
                weights = np.ones(k) / k
            else:
                # Exponential weighting favoring better points
                weights = np.exp(-3.0 * (top_vals - min_val) / (max_val - min_val + 1e-10))
                weights /= np.sum(weights)
            
            # Weighted centroid
            centroid = np.sum(top_pts * weights[:, np.newaxis], axis=0)
            
            # Covariance of top points (weighted)
            centered = top_pts - centroid
            cov = (centered * weights[:, np.newaxis]).T @ centered
            cov = (cov + cov.T) / 2  # symmetrize
            
            # Eigendecomposition to find principal direction
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
                # The ridge direction is the one with largest variance
                ridge_dir = eigvecs[:, -1]
                # Ensure consistent orientation
                if ridge_dir[0] < 0:
                    ridge_dir = -ridge_dir
                return ridge_dir
            except Exception:
                return None

        # Phase 3: Ridge-aligned local search
        if calls < budget and elapsed() < deadline:
            ridge_dir = estimate_ridge_direction()
            
            if ridge_dir is not None:
                # Project best point onto ridge and do 1D search along ridge
                # Start from best_x and move along ridge direction
                x_curr = best_x.copy()
                f_curr = best_f
                sigma_ridge = 0.5
                
                for _ in range(min(100, budget - calls)):
                    if calls >= budget or elapsed() > deadline:
                        break
                    
                    # Try moving along ridge direction
                    x_mut_ridge = x_curr + sigma_ridge * ridge_dir
                    x_mut_ridge = np.clip(x_mut_ridge, -5.0, 5.0)
                    f_mut_ridge = safe_call(x_mut_ridge)
                    
                    # Try moving opposite
                    x_mut_ridge_neg = x_curr - sigma_ridge * ridge_dir
                    x_mut_ridge_neg = np.clip(x_mut_ridge_neg, -5.0, 5.0)
                    f_mut_ridge_neg = safe_call(x_mut_ridge_neg)
                    
                    if f_mut_ridge < f_curr:
                        x_curr = x_mut_ridge.copy()
                        f_curr = f_mut_ridge
                        sigma_ridge *= 1.2
                    elif f_mut_ridge_neg < f_curr:
                        x_curr = x_mut_ridge_neg.copy()
                        f_curr = f_mut_ridge_neg
                        sigma_ridge *= 1.2
                    else:
                        sigma_ridge *= 0.7
                    
                    if f_curr < best_f:
                        best_f = f_curr
                        best_x = x_curr.copy()
            
            # Phase 3b: Standard adaptive (1+1)-ES for remaining budget
            if calls < budget and elapsed() < deadline:
                sigma = 0.5
                x_curr = best_x.copy()
                f_curr = best_f
                success_count = 0
                window = 8
                stall = 0
                max_stall = 30

                while calls < budget and elapsed() < deadline:
                    z = np.random.default_rng().standard_normal(n)
                    x_mut = x_curr + sigma * z
                    x_mut = np.clip(x_mut, -5.0, 5.0)
                    f_mut = safe_call(x_mut)

                    if f_mut < f_curr:
                        x_curr = x_mut.copy()
                        f_curr = f_mut
                        success_count += 1
                        stall = 0
                    else:
                        stall += 1

                    if success_count >= window:
                        sigma *= 1.4
                        success_count = 0
                    elif success_count <= window // 3:
                        sigma *= 0.7
                        success_count = 0

                    if stall >= max_stall:
                        # Restart from best with perturbation
                        x_curr = best_x + np.random.default_rng().standard_normal(n) * 1.5
                        x_curr = np.clip(x_curr, -5.0, 5.0)
                        f_curr = safe_call(x_curr)
                        sigma = 0.5
                        stall = 0
                        success_count = 0

                    if f_curr < best_f:
                        best_f = f_curr
                        best_x = x_curr.copy()

        # Phase 4: Final local polish using Nelder-Mead
        if calls < budget and elapsed() < deadline:
            try:
                from scipy.optimize import minimize
                def obj(x):
                    nonlocal calls
                    if calls >= budget or elapsed() > deadline:
                        return best_f
                    return safe_call(x)
                
                res = minimize(obj, best_x, method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10,
                                        'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
