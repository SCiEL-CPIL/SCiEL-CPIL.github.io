"""Rewritten optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(n))
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n**2))
        
        # State Initialization
        rng = np.random.default_rng(42)
        x_mean = rng.uniform(-2, 2, n)
        C = np.eye(n)
        D = np.ones(n)
        B = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        sigma = 1.0
        
        best_x = x_mean.copy()
        best_fx = func(best_x)
        calls = 1
        
        while calls < budget:
            # Update Cholesky Decomposition of C
            try:
                # Ensure C is SPD
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-12:
                    C += (1e-12 - np.min(eigvals)) * np.eye(n)
                L = np.linalg.cholesky(C)
                D = np.diag(L)
                B = np.linalg.inv(L).T
            except:
                # Fallback to identity if C becomes singular
                C = np.eye(n)
                D = np.ones(n)
                B = np.eye(n)
                
            # Determine batch size
            batch_size = min(lambda_, budget - calls)
            
            # Generate Offspring
            Z = rng.standard_normal((batch_size, n))
            X = x_mean + sigma * ((Z * D) @ B.T)
            
            # Handle Boundaries via Reflection
            for i in range(batch_size):
                for j in range(n):
                    if X[i, j] < -5.0:
                        X[i, j] = -10.0 - X[i, j]
                    elif X[i, j] > 5.0:
                        X[i, j] = 10.0 - X[i, j]
                    X[i, j] = np.clip(X[i, j], -5.0, 5.0)
                    
            # Evaluate
            F = np.array([func(x) for x in X])
            calls += batch_size
            
            # Update Best
            min_idx = np.argmin(F)
            if F[min_idx] < best_fx:
                best_fx = F[min_idx]
                best_x = X[min_idx].copy()
                
            # Sort and Select
            idx = np.argsort(F)
            mu_idx = idx[:mu]
            X_w = X[mu_idx]
            
            # Update Mean
            x_new = np.sum(weights[:, np.newaxis] * X_w, axis=0)
            
            # Update Paths and Step Size
            sigma_old = sigma
            y = (x_new - x_mean) / sigma
            
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (B.T @ (y / D))
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp(cs / damps * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)
            
            p_c = (1 - cc) * p_c + cc * np.sqrt(mueff) * (B.T @ (y / D))
            
            # Update Covariance Matrix
            diff = X_w - x_mean
            H = np.sum(weights[:, np.newaxis, np.newaxis] * np.einsum('ij,ik->jk', diff, diff), axis=0)
            
            delta_sigma_term = 1 - (sigma_old / sigma)**2
            # Clamp delta_sigma_term to avoid instability if sigma changes drastically
            delta_sigma_term = np.clip(delta_sigma_term, -1.0, 1.0)
            
            C_new = (1 - c1 - cmu) * C
            C_new += c1 * (np.outer(p_c, p_c) + delta_sigma_term * np.outer(p_sigma, p_sigma) / (chi_n**2))
            C_new += cmu * H
            
            # Enforce Symmetry
            C_new = (C_new + C_new.T) / 2
            
            # Ensure PSD
            eigvals = np.linalg.eigvalsh(C_new)
            if np.min(eigvals) < 1e-12:
                C_new += (1e-12 - np.min(eigvals)) * np.eye(n)
                
            C = C_new
            x_mean = x_new
            
        return best_x, best_fx
# EVOLVE-BLOCK-END
