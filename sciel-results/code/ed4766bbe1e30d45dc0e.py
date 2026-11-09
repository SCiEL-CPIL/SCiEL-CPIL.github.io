"""Surrogate-assisted optimizer using Gaussian Process with Expected Improvement for Rastrigin rotated."""
import numpy as np
import time
import math
from scipy.stats import norm


class Optimizer:
    """Surrogate-assisted optimizer for Rastrigin rotated in 5D."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial random sampling to build initial model
        n_init = 40
        X = np.zeros((0, dim))
        Y = np.array([])
        best_x = None
        best_f = np.inf

        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            X = np.vstack([X, x]) if X.size > 0 else x.reshape(1, -1)
            Y = np.append(Y, fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            return (np.zeros(dim), 0.0)

        calls_used = n_init

        # GP hyperparameters
        # Use a simple ARD (Automatic Relevance Determination) kernel
        # K(x, x') = sigma_f^2 * exp(-0.5 * sum((x_i - x'_i)^2 / l_i^2)) + sigma_n^2 * I

        def compute_kernel(X1, X2, length_scales, sigma_f, sigma_n):
            """Compute GP kernel matrix between X1 and X2."""
            # X1: (n1, dim), X2: (n2, dim)
            # Compute squared distances with ARD
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]  # (n1, n2, dim)
            sq_dist = np.sum(diff ** 2 / length_scales[np.newaxis, np.newaxis, :], axis=2)  # (n1, n2)
            K = sigma_f ** 2 * np.exp(-0.5 * sq_dist)
            if X1.shape[0] == X2.shape[0] and X1.shape[0] == X2.shape[1]:
                K += sigma_n ** 2 * np.eye(X1.shape[0])
            return K

        def predict_gp(X_new, X_train, Y_train, length_scales, sigma_f, sigma_n):
            """Predict mean and variance at X_new using GP."""
            K = compute_kernel(X_train, X_train, length_scales, sigma_f, sigma_n)
            K_new = compute_kernel(X_train, X_new, length_scales, sigma_f, 0)
            try:
                K_inv = np.linalg.inv(K)
            except:
                K_inv = np.linalg.pinv(K)
            mean = K_new.T @ K_inv @ Y_train
            v = compute_kernel(X_new, X_new, length_scales, sigma_f, 0)
            var = np.diag(v - K_new @ K_inv @ K_new.T)
            var = np.maximum(var, 1e-10)
            return mean, var

        def expected_improvement(mean, var, best_f):
            """Compute Expected Improvement acquisition function."""
            sigma = np.sqrt(var)
            z = (best_f - mean) / sigma
            ei = (best_f - mean) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        # Initialize hyperparameters
        length_scales = np.ones(dim) * 1.5
        sigma_f = 1.0
        sigma_n = 1e-5

        # Phase 2: Surrogate-assisted optimization loop
        max_iter = 200
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break

            # Predict mean and variance on a candidate set
            # Use a mix of random points and points near current best
            n_candidates = 200
            candidates = np.zeros((n_candidates, dim))
            
            # 50% random, 50% local around best
            n_random = n_candidates // 2
            n_local = n_candidates - n_random
            
            candidates[:n_random] = rng.uniform(-5, 5, (n_random, dim))
            
            # Local exploration around best with decreasing radius
            radius = max(0.1, 1.0 - it / max_iter * 0.9)
            candidates[n_random:] = best_x + rng.normal(0, radius, (n_local, dim))
            candidates = np.clip(candidates, -5, 5)

            # Evaluate acquisition function
            try:
                mean, var = predict_gp(candidates, X, Y, length_scales, sigma_f, sigma_n)
                ei = expected_improvement(mean, var, best_f)
            except:
                # Fallback: just pick random
                ei = rng.random(n_candidates)

            # Select best candidate
            idx = np.argmax(ei)
            x_new = candidates[idx]

            # Evaluate
            fx_new = func(x_new)
            calls_used += 1

            # Update data
            X = np.vstack([X, x_new])
            Y = np.append(Y, fx_new)

            # Update best
            if fx_new < best_f:
                best_f = fx_new
                best_x = x_new.copy()

            # Occasionally adapt hyperparameters (simple heuristic)
            if it % 20 == 0 and it > 0:
                # Simple adaptation: adjust length scales based on spread of data
                if X.shape[0] > 10:
                    # Estimate typical distance between good points
                    # For simplicity, keep length scales fixed or do minor adjustment
                    pass

        return (best_x, best_f)
