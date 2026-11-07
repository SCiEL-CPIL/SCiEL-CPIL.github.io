"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.spatial.distance import cdist

class Optimizer:
    """Bayesian Optimization with Gaussian Process surrogate."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280
        rng = np.random.default_rng(42)
        
        # Initialize with a small set of points to build the GP
        X = np.zeros((0, self.dim))
        Y = np.zeros((0,))
        best_x = None
        best_f = np.inf
        
        # 1. Initial random sampling to explore the space
        n_init = 10
        for _ in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            X = np.vstack([X, x])
            Y = np.append(Y, fx)
            if fx < best_f:
                best_f = fx
                best_x = x

        # GP Hyperparameters
        length_scales = np.ones(self.dim) * 2.0
        noise = 1e-8
        signal_var = 1.0
        
        # 2. Bayesian Optimization Loop
        for i in range(self.budget - n_init):
            if time.time() > deadline:
                break
            
            # Generate candidate points
            n_candidates = 50
            X_cand = rng.uniform(-5, 5, (n_candidates, self.dim))
            
            # Compute GP predictions
            # Kernel: Matern 3/2 or RBF. RBF is faster to compute.
            # K(x, x') = signal_var * exp(-0.5 * ||x - x'||^2 / l^2)
            
            # Efficient distance calculation
            # X is (n, d), X_cand is (n_cand, d)
            # dists is (n_cand, n)
            dists = cdist(X_cand, X, metric='sqeuclidean')
            
            # Apply length scales
            # We need to sum over dimensions: sum_d (x_i,d - x_j,d)^2 / l_d^2
            # cdist gives sum of squared diffs. We need weighted sum.
            # For simplicity and speed, if we assume isotropic or just use raw sqeuclidean
            # with a single effective length scale, it's faster.
            # Given dim=5, we can compute weighted distances manually or vectorized.
            
            # Let's do vectorized weighted distance
            # X_cand: (n_cand, d), X: (n, d)
            # diff: (n_cand, n, d) -- might be memory heavy if n is large, but n <= 500, n_cand=50.
            # 50 * 500 * 5 = 125,000 floats. Safe.
            diff = X_cand[:, np.newaxis, :] - X[np.newaxis, :, :]
            # Weighted squared distance
            w_sq_dists = np.sum(diff**2 / (length_scales**2), axis=2)
            
            K = signal_var * np.exp(-0.5 * w_sq_dists)
            K_diag = np.diag(K)
            
            # Prior mean is 0. 
            # We need to solve for GP parameters.
            # To make it robust and fast, we'll use a simplified update or just standard GP prediction.
            # Standard GP prediction:
            # m(x*) = K(x*, X) @ alpha
            # alpha = (K(X, X) + noise*I)^-1 (Y - m(X))
            
            # Since we don't have a closed form for alpha without inverting K(X,X) every time,
            # and inverting a 500x500 matrix 500 times is expensive (500 * 500^3 is too slow? 
            # 500 * 125e6 = 62.5e9 ops. Too slow in Python/NumPy without optimization).
            
            # Alternative: Use a simpler surrogate or optimize the GP update.
            # Actually, 500 inversions of 500x500 matrices might be borderline.
            # Let's check: np.linalg.inv(500x500) takes ~1ms-5ms in NumPy.
            # 500 * 5ms = 2.5s. This is acceptable within 300s limit.
            
            K_XX = signal_var * np.exp(-0.5 * np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :])**2 / (length_scales**2), axis=2))
            K_XX += noise * np.eye(len(X))
            
            # Solve for alpha
            try:
                alpha = np.linalg.solve(K_XX, Y)
            except np.linalg.LinAlgError:
                # Fallback if singular
                alpha = np.linalg.lstsq(K_XX, Y, rcond=None)[0]
            
            # Predict mean and variance for candidates
            K_star = K # (n_cand, n)
            mu = K_star @ alpha
            # Variance: K(x*, x*) - K_star @ K_XX^-1 @ K_star.T
            # We can reuse the Cholesky factorization from solve if we did it, 
            # but np.linalg.solve doesn't expose it.
            # Let's use Cholesky for stability and speed in subsequent steps if we cached it, 
            # but here we recompute.
            try:
                L = np.linalg.cholesky(K_XX)
                # Solve L v = K_star.T
                v = np.linalg.solve(L, K_star.T)
                # v shape (n, n_cand)
                var = K_diag - np.sum(v**2, axis=0)
                var = np.maximum(var, 1e-8)
            except np.linalg.LinAlgError:
                # Fallback variance estimation
                var = np.full(n_candidates, signal_var)
            
            std = np.sqrt(var)
            
            # Expected Improvement
            # EI = (mu_best - mu) * Phi((mu_best - mu)/std) + std * phi((mu_best - mu)/std)
            # where mu_best is the current best (lowest) value.
            # Note: For minimization, we flip signs or adjust formula.
            # Standard EI for minimization:
            # EI(x) = (f_min - mu(x)) * Phi((f_min - mu(x))/sigma(x)) + sigma(x) * phi((f_min - mu(x))/sigma(x))
            
            z = (best_f - mu) / std
            EI = (best_f - mu) * norm.cdf(z) + std * norm.pdf(z)
            
            # Select candidate with max EI
            idx = np.argmax(EI)
            x_next = X_cand[idx]
            
            # Evaluate
            fx_next = func(x_next)
            
            # Update data
            X = np.vstack([X, x_next])
            Y = np.append(Y, fx_next)
            
            if fx_next < best_f:
                best_f = fx_next
                best_x = x_next
                
            # Optional: Update hyperparameters (length scales) periodically
            # For simplicity and speed, we keep them fixed or do a very cheap update.
            # Given the "Bent Cigar" is ill-conditioned, fixed isotropic LS might be suboptimal.
            # However, adapting LS requires optimization (L-BFGS-B) which is slow.
            # We can try to adapt LS based on the spread of Y vs X.
            # For now, fixed LS=2.0 is a reasonable start for [-5,5]^5.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
