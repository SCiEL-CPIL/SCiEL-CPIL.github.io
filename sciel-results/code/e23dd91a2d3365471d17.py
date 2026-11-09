"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES based continuous optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lambda_pop = 4 + int(3 * np.log(n))
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
        
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        C_sqrt = np.eye(n)
        C_sqrt_inv = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        gen = 0
        best_x = mean.copy()
        best_f = float('inf')
        evals_used = 0
        
        def update_chol(C):
            try:
                L = np.linalg.cholesky(C)
                C_sqrt = L.T
                C_sqrt_inv = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                C_sqrt_inv = np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            return C_sqrt, C_sqrt_inv

        C_sqrt, C_sqrt_inv = update_chol(C)
        
        while evals_used + lambda_pop <= budget:
            Z = np.random.randn(lambda_pop, n)
            X = mean + sigma * (Z @ C_sqrt.T)
            X = np.clip(X, -5.0, 5.0)
            
            F = np.array([func(x) for x in X])
            evals_used += lambda_pop
            gen += 1
            
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()
                
            order = np.argsort(F)
            X_sorted = X[order]
            
            x_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            z = C_sqrt_inv @ (x_new - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** gen)) < (1 + 2 / (n + 1)) * chi_n
            sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - mean) / sigma
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            
            C = (1 + c_1 * (2 - c_1 + delta_hsig) - c_mu) * C
            C += c_1 * np.outer(p_c, p_c)
            C += c_mu * np.sum(weights[:, None, None] * np.outer(X_sorted[:mu] - mean, X_sorted[:mu] - mean) / sigma ** 2, axis=0)
            
            C_sqrt, C_sqrt_inv = update_chol(C)
            mean = x_new
            
        # Use remaining budget for random search if any
        while evals_used < budget:
            x = np.random.uniform(-5.0, 5.0, size=n)
            f = func(x)
            evals_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
        return best_x, best_f
# EVOLVE-BLOCK-END
