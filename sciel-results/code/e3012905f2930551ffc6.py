"""Rewritten program: CMA-ES style optimizer for BBOB f3 (Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style continuous optimizer for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Population size: lambda = 4 + floor(3 * ln(n))
        lam = max(4 + int(3 * np.log(n)), 10)
        
        # Initial parameters
        # Start near center, with moderate sigma
        mean = np.zeros(n)
        sigma = 1.5
        
        # Precompute weights
        mu = lam // 2  # number of parents
        # Weights: w_i = ln(mu+1) - ln(i+1) for i=0..mu-1
        weights = np.array([np.log(mu + 1) - np.log(i + 1) for i in range(mu)])
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Step size control parameters
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # Covariance matrix
        C = np.eye(n)
        # Eigen decomposition cache
        C_eigenvalues = np.ones(n)
        C_eigenvectors = np.eye(n)
        
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Hessian update history for cumulation
        h_sigma = 1.0
        
        best_x = mean.copy()
        best_f = func(mean)
        
        # Track best solution
        for _ in range(budget):
            # Generate population
            population = np.zeros((lam, n))
            for i in range(lam):
                # Sample from N(mean, sigma^2 * C)
                # Use Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to eigenvalue decomposition
                    L = C_eigenvectors @ np.diag(np.sqrt(np.maximum(C_eigenvalues, 0)))
                
                z = rng.standard_normal(n)
                population[i] = mean + sigma * (L @ z)
            
            # Evaluate population
            fitness = np.array([func(population[i]) for i in range(lam)])
            
            # Track best
            idx_min = np.argmin(fitness)
            if fitness[idx_min] < best_f:
                best_f = fitness[idx_min]
                best_x = population[idx_min].copy()
            
            # Sort by fitness
            idx_sorted = np.argsort(fitness)
            sorted_pop = population[idx_sorted]
            sorted_fit = fitness[idx_sorted]
            
            # Update mean
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * sorted_pop[:mu], axis=0)
            
            # Update evolution path for covariance matrix
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (mean - old_mean)
            
            # Update evolution path for sigma
            # Need to compute C^{-1/2} (mean - old_mean)
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
            except np.linalg.LinAlgError:
                C_inv_sqrt = C_eigenvectors @ np.diag(1.0 / np.sqrt(np.maximum(C_eigenvalues, 1e-10))) @ C_eigenvectors.T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (mean - old_mean)
            
            # Update sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            # Heuristic for h_sigma
            h_sigma = 1.0 if norm_p_sigma < (1.4 + 1 / c_sigma) * chi_n else 0.0
            sigma *= np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update covariance matrix
            delta_sigma = (1 - h_sigma) * c_sigma * (2 - c_sigma)
            rank_one = np.outer(mean - old_mean, mean - old_mean)
            rank_mu = np.sum(weights[:, None, None] * np.outer(sorted_pop[:mu] - old_mean, sorted_pop[:mu] - old_mean), axis=0)
            
            C = (1 + c_1 * (2 - c_1) - c_mu) * C + c_1 * (n + 1 - c_c) * np.outer(p_c, p_c) + c_mu * rank_mu - delta_sigma * sigma ** 2 * C
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Regularize to maintain positive semi-definiteness
            # Add small identity if needed
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C += (1e-10 - np.min(eigvals)) * np.eye(n)
            
            # Update eigen decomposition periodically (expensive)
            # For simplicity, update every 10 generations or when needed
            # Here we'll just ensure C is valid for next iteration
            
            # If budget is exhausted, break
            if _ == budget - 1:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
