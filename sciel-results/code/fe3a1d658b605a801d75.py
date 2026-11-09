"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimization with Matern-5/2 GP for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _matern52(self, X, Y, lengthscales, sigma_f, noise_var):
        """Compute covariance matrix using Matern-5/2 kernel."""
        # X, Y: arrays of shape (n, d)
        # Compute pairwise distances
        diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        
        # Matern-5/2 kernel
        sqrt3_dist = np.sqrt(3) * dist / lengthscales
        k = sigma_f ** 2 * (1 + sqrt3_dist + sqrt3_dist ** 2 / 3) * np.exp(-sqrt3_dist)
        
        # Add noise to diagonal for Y
        if X is Y:
            k += noise_var * np.eye(k.shape[0])
        
        return k

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        # Phase 1: Initial random sampling (10% of budget)
        n_init = max(10, budget // 10)
        X = []
        Y = []
        
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            X.append(x)
            Y.append(fx)
        
        X = np.array(X)
        Y = np.array(Y)
        
        best_x = X[np.argmin(Y)].copy()
        best_f = Y.min()

        # GP hyperparameters (tuned for Weierstrass-like functions)
        lengthscales = np.ones(n) * 1.0
        sigma_f = 1.0
        noise_var = 1e-8

        # Phase 2: Bayesian Optimization loop
        n_bo = budget - n_init
        for i in range(n_bo):
            if time.time() > deadline:
                break
            
            # Build GP model
            K = self._matern52(X, X, lengthscales, sigma_f, noise_var)
            
            # Compute mean and variance for acquisition function
            # Use Expected Improvement (EI)
            mu_min = Y.min()
            alpha = 0.01  # Exploration parameter
            
            # Generate candidate points for acquisition maximization
            n_candidates = 200
            X_cand = rng.uniform(-5, 5, (n_candidates, n))
            
            # Compute GP posterior for candidates
            K_s = self._matern52(X_cand, X, lengthscales, sigma_f, 0)
            K_ss = self._matern52(X_cand, X_cand, lengthscales, sigma_f, 0)
            
            try:
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                K_inv = np.linalg.pinv(K)
            
            # Posterior mean and variance
            mu = K_s @ K_inv @ Y
            v = K_s @ K_inv @ K_s.T
            var = np.diag(K_ss) - np.diag(v)
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)
            
            # Expected Improvement
            z = (mu_min - mu - alpha) / std
            EI = std * (z * norm.cdf(z) + norm.pdf(z))
            EI = np.where(std > 1e-10, EI, 0)
            
            # Select best candidate
            idx = np.argmax(EI)
            x_new = X_cand[idx]
            fx_new = func(x_new)
            
            # Update data
            X = np.vstack([X, x_new])
            Y = np.append(Y, fx_new)
            
            if fx_new < best_f:
                best_f = fx_new
                best_x = x_new.copy()
            
            # Adapt hyperparameters occasionally
            if i % 50 == 49 and i > 0:
                # Simple adaptation: adjust lengthscales based on recent improvements
                recent_Y = Y[-20:] if len(Y) >= 20 else Y
                if np.std(recent_Y) < 0.1:
                    lengthscales *= 0.8  # Shrink search when converged
                else:
                    lengthscales = np.clip(lengthscales * 1.1, 0.1, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
