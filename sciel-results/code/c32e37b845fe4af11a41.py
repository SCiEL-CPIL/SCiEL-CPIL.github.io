"""Rewritten optimizer using a proper CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Optimizer using CMA-ES for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        # Weights
        weights = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam + 1))
        weights /= np.sum(weights)
        mu = int(np.sum(weights > 0))
        mueff = 1 / np.sum(weights[:mu] ** 2)
        
        # Initial step size
        sigma = 2.0
        
        # Initial mean: Use a few random points to pick a good start
        # Spend ~10% of budget on initialization
        n_init = min(10, budget // 10)
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def eval_point(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return x, f

        # Phase 1: Initialization
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            x, fx = eval_point(x)

        mean = best_x.copy()
        
        # Covariance matrix
        C = np.eye(dim)
        # Inverse square root of C (initially identity)
        C_isqrt = np.eye(dim)
        # Eigen decomposition cache
        eigvals = np.ones(dim)
        eigvecs = np.eye(dim)
        
        # Adaptation parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff - 2 + 1 / mueff) / (dim + 2) ** 2 - 1))
        
        # Memory
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Main Loop
        while calls_made < budget:
            # Generate population
            offspring = []
            for i in range(lam):
                if calls_made >= budget:
                    break
                # Sample
                z = rng.standard_normal(dim)
                # Transform using Cholesky decomposition of C
                # We maintain C_isqrt. The sample is mean + sigma * C_isqrt @ z
                # But C changes, so we need to update C_isqrt.
                # For efficiency, we can use the eigen decomposition:
                # C = eigvecs @ diag(eigvals) @ eigvecs.T
                # C_isqrt = eigvecs @ diag(1/sqrt(eigvals)) @ eigvecs.T
                # This is expensive to compute every step.
                # Standard CMA-ES uses Cholesky.
                
                # Let's stick to a simpler, robust variant:
                # We will update C via eigen decomposition periodically or use the rank-1/rank-mu update and re-decompose.
                
                y = mean + sigma * (eigvecs @ (eigvals ** -0.5) @ (eigvecs.T @ z))
                y, fy = eval_point(y)
                offspring.append((fy, y))
            
            if calls_made >= budget:
                break
            
            # Sort offspring
            offspring.sort(key=lambda t: t[0])
            
            # Update mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * offspring[i][1]
            
            # Calculate normalized step
            y_vec = (new_mean - mean) / sigma
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_isqrt @ y_vec
            
            # Check h_sigma
            h_sigma = np.sum(p_sigma ** 2) / (1 - (1 - c_sigma) ** (2 * (calls_made - n_init + 1))) < (1 + 2 / (dim + 1)) * (1 - c_sigma)
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y_vec
            
            # Update covariance matrix C
            # Rank-1 update
            rank1 = np.outer(p_c, p_c)
            # Rank-mu update
            rankmu = np.zeros((dim, dim))
            for i in range(mu):
                wi = weights[i]
                yi = (offspring[i][1] - mean) / sigma
                rankmu += wi * np.outer(yi, yi)
            
            C = (1 + c_1 * h_sigma * (2 - c_c) - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rankmu
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(chi_n) - 1))
            # chi_n is the expected norm of a standard normal vector
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            # Update mean
            mean = new_mean
            
            # Update C_isqrt via eigen decomposition
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues
            eigvals = np.clip(eigvals, 1e-12, None)
            C_isqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Check for stagnation
            # If sigma is too small or mean hasn't changed much, restart?
            # For now, we rely on the adaptation.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
