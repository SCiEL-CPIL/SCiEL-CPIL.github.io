"""Bayesian Optimizer for BBOB f22 using Gaussian Process surrogate."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.stats import norm

warnings.filterwarnings('ignore')

class Optimizer:
    """Bayesian Optimizer for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280.0  # Safety margin
        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0
        n = self.dim

        def check_time():
            return (time.time() - start_time) < time_limit

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if not check_time():
                return np.inf
            calls_made += 1
            try:
                fx = func(x)
            except Exception:
                return np.inf
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx if np.isfinite(fx) else np.inf

        rng = np.random.default_rng(42)

        # Initialize with LHS
        n_init = 20
        X = np.zeros((n_init, n))
        for i in range(n):
            perm = rng.permutation(n_init)
            u = (perm + 0.5) / n_init
            X[:, i] = -5 + 10 * u
        
        Y = np.array([safe_func(x) for x in X])
        
        # Filter out inf values for GP fitting
        valid_mask = np.isfinite(Y)
        X_valid = X[valid_mask]
        Y_valid = Y[valid_mask]
        
        # Hyperparameters for GP
        length_scales = np.ones(n) * 1.0
        sigma_f = 1.0
        sigma_n = 1e-8

        def rbf_kernel(X1, X2, length_scales, sigma_f):
            X1_norm = np.sum(X1**2, axis=1)
            X2_norm = np.sum(X2**2, axis=1)
            dist_sq = X1_norm[:, None] + X2_norm[None, :] - 2 * np.dot(X1, X2.T)
            dist_sq = np.maximum(dist_sq, 0)
            dist_sq /= (length_scales ** 2)
            return sigma_f**2 * np.exp(-0.5 * dist_sq)

        def compute_ei(mu, sigma, best_f_val):
            sigma_safe = np.where(sigma < 1e-10, 1e-10, sigma)
            z = (best_f_val - mu) / sigma_safe
            ei = (best_f_val - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
            return np.maximum(ei, 0)

        max_gp_points = 100
        
        while calls_made < self.budget and check_time():
            if len(X_valid) > max_gp_points:
                X_gp = X_valid[-max_gp_points:]
                Y_gp = Y_valid[-max_gp_points:]
            else:
                X_gp = X_valid
                Y_gp = Y_valid
            
            if len(X_gp) < 2:
                x_new = rng.uniform(-5, 5, size=n)
                f_new = safe_func(x_new)
                if np.isfinite(f_new):
                    X = np.vstack([X, x_new])
                    Y = np.append(Y, f_new)
                    valid_mask = np.isfinite(Y)
                    X_valid = X[valid_mask]
                    Y_valid = Y[valid_mask]
                continue

            K = rbf_kernel(X_gp, X_gp, length_scales, sigma_f)
            K += sigma_n * np.eye(len(X_gp))
            
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-6 * np.eye(len(X_gp))
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    x_new = rng.uniform(-5, 5, size=n)
                    f_new = safe_func(x_new)
                    if np.isfinite(f_new):
                        X = np.vstack([X, x_new])
                        Y = np.append(Y, f_new)
                        valid_mask = np.isfinite(Y)
                        X_valid = X[valid_mask]
                        Y_valid = Y[valid_mask]
                    continue

            n_candidates = 200
            X_cand = rng.uniform(-5, 5, size=(n_candidates, n))
            
            K_s = rbf_kernel(X_cand, X_gp, length_scales, sigma_f)
            
            v = np.linalg.solve(L.T, Y_gp)
            alpha = np.linalg.solve(L, v)
            
            mu = K_s @ alpha
            v2 = np.linalg.solve(L, K_s.T)
            sigma_sq = sigma_f**2 - np.sum(v2**2, axis=0)
            sigma_sq = np.maximum(sigma_sq, 1e-10)
            sigma = np.sqrt(sigma_sq)
            
            ei = compute_ei(mu, sigma, best_f)
            
            idx_best = np.argmax(ei)
            x_new = X_cand[idx_best]
            
            f_new = safe_func(x_new)
            
            X = np.vstack([X, x_new])
            Y = np.append(Y, f_new)
            valid_mask = np.isfinite(Y)
            X_valid = X[valid_mask]
            Y_valid = Y[valid_mask]

        return best_x, best_f

# EVOLVE-BLOCK-END
