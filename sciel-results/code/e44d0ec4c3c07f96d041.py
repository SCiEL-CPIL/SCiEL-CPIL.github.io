"""Bayesian-Optimization-inspired surrogate optimizer for BBOB f6."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """Gaussian Process surrogate + Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget

        # Initial points: center + random
        X = [np.zeros(dim)]
        for _ in range(min(10, budget - 1)):
            X.append(rng.uniform(-4, 4, dim))
        X = np.array(X)

        Y = np.array([func(x) for x in X])
        best_idx = np.argmin(Y)
        best_x = X[best_idx].copy()
        best_y = Y[best_idx]

        # Hyperparameters for GP
        # We use a simple squared-exponential kernel with automatic relevance determination
        # To keep it fast, we fix lengthscales or use a simple isotropic one
        # Let's use isotropic lengthscale l=1.0, signal variance s2=1.0, noise variance n2=1e-10
        # Actually, let's try to adapt lengthscale slightly or just keep it simple.
        # Given dim=5, N grows up to 500. GP computation is O(N^3).
        # 500^3 = 125,000,000 operations. This might be too slow if done naively in Python loops,
        # but with NumPy vectorization it should be fine.
        # However, we need to solve linear systems for each candidate point.
        
        # Strategy:
        # 1. Fit GP (compute K, L, alpha)
        # 2. Generate candidates (e.g., 100-200 points)
        # 3. Compute EI for all candidates
        # 4. Pick best candidate
        # 5. Evaluate, update X, Y
        # 6. Repeat

        # To ensure speed, we limit the number of GP iterations if time runs out.
        # Each GP fit with N points takes O(N^3).
        # For N=500, this is manageable.
        
        # Kernel parameters
        l = 1.0  # lengthscale
        s2 = 1.0 # signal variance
        n2 = 1e-10 # noise variance

        # Normalization of Y to improve GP conditioning
        y_min = Y.min()
        y_max = Y.max()
        y_range = y_max - y_min
        if y_range < 1e-10:
            y_range = 1.0
        Y_norm = (Y - y_min) / y_range

        for i in range(len(X), budget):
            if time.time() > deadline - 10.0:
                # Not enough time for a full GP cycle, break or do simple random?
                # Break to return best found so far
                break

            # Compute GP parameters
            # K = s2 * exp(-0.5 * ||x_i - x_j||^2 / l^2) + n2 * I
            # Precompute distances
            # Use vectorized distance calculation
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=2)
            K = s2 * np.exp(-0.5 * sq_dist / (l ** 2))
            K[np.diag_indices_from(K)] += n2
            
            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Add jitter if singular
                K[np.diag_indices_from(K)] += 1e-6
                L = np.linalg.cholesky(K)
            
            # Solve K alpha = Y_norm
            # L L^T alpha = Y_norm
            # L v = Y_norm -> v
            v = np.linalg.solve(L, Y_norm)
            # L^T alpha = v -> alpha
            alpha = np.linalg.solve(L.T, v)

            # Generate candidate points
            # Mix of random points and points near current best
            n_candidates = 200
            candidates = rng.uniform(-5, 5, (n_candidates, dim))
            # Add some points around best_x to encourage exploitation
            for _ in range(50):
                candidates[_ % n_candidates] = np.clip(best_x + rng.normal(0, 0.5, dim), -5, 5)

            # Compute GP mean and variance for candidates
            # k* = s2 * exp(-0.5 * ||x_c - x_i||^2 / l^2)
            # mean = k*^T alpha
            # var = s2 - k*^T L^{-T} L^{-1} k* = s2 - k*^T alpha
            # Note: var = s2 - k*^T alpha is only true if we use the same K structure.
            # Actually, var(x) = s2 - k(x, X)^T alpha is correct for the posterior variance.
            
            # Compute distances from candidates to X
            # candidates: (n_c, dim), X: (N, dim)
            diff_c = candidates[:, np.newaxis, :] - X[np.newaxis, :, :]
            sq_dist_c = np.sum(diff_c ** 2, axis=2)
            k_star = s2 * np.exp(-0.5 * sq_dist_c / (l ** 2))
            
            # Mean
            mean = k_star @ alpha
            
            # Variance
            var = s2 - k_star @ alpha
            # Clip variance to be positive (numerical issues)
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)

            # Expected Improvement (EI)
            # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
            # where z = (f_best - mu - xi) / sigma
            # We want to minimize, so f_best is the current best (lowest) value.
            # EI = (best_y_norm - mu) * Phi((best_y_norm - mu) / sigma) + sigma * phi((best_y_norm - mu) / sigma)
            # Note: best_y_norm is the normalized best value.
            
            best_y_norm = Y_norm[best_idx]
            
            z = (best_y_norm - mean) / std
            # Handle z = 0 case (sigma=0)
            z = np.where(std > 1e-10, z, 0)
            
            ei = (best_y_norm - mean) * norm.cdf(z) + std * norm.pdf(z)
            
            # Select best candidate
            idx_best = np.argmax(ei)
            x_new = candidates[idx_best]
            
            # Evaluate
            y_new = func(x_new)
            
            # Update data
            X = np.vstack([X, x_new])
            Y = np.vstack([Y, y_new])
            Y_norm = (Y - y_min) / y_range
            
            if y_new < best_y:
                best_y = y_new
                best_x = x_new.copy()
                best_idx = len(X) - 1

        return best_x, best_y
# EVOLVE-BLOCK-END
