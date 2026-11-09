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
        lam = 10 
        mu = lam // 2
        
        # Weights
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= weights.sum()
        
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff)))
        
        # Initial state
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        C_cholesky = np.eye(n)
        
        # Path accumulators
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        evals_used = 0
        best_x = np.zeros(n)
        best_f = float('inf')
        
        rng = np.random.default_rng(42)
        
        while evals_used < budget:
            pop = np.zeros((lam, n))
            fits = np.zeros(lam)
            
            for i in range(lam):
                if evals_used >= budget:
                    break
                z = rng.standard_normal(n)
                x = x_mean + sigma * (C_cholesky @ z)
                x = np.clip(x, -5.0, 5.0)
                pop[i] = x
                fits[i] = func(x)
                evals_used += 1
                
                if fits[i] < best_f:
                    best_f = fits[i]
                    best_x = x.copy()
            
            if evals_used >= budget:
                break
            
            order = np.argsort(fits)
            pop = pop[order]
            fits = fits[order]
            
            x_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * pop[:mu], axis=0)
            
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ ((x_mean - x_old) / sigma))
            
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (evals_used / lam + 1))) < (1.4 + 2 / (n + 1)) * chi_n
            
            if hsig:
                sigma *= np.exp(cs / 5 * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            p_c = (1 - cs) * p_c + hsig * np.sqrt(cs * (2 - cs) * mueff) * (x_mean - x_old) / sigma
            
            diff = (pop[:mu] - x_old) / sigma
            rank_mu = np.sum(weights[:, None, None] * (diff[:, :, None] * diff[:, None, :]), axis=0)
            
            C_new = (1 - c1 - cmu) * C
            C_new += c1 * (np.outer(p_c, p_c) + (1 - hsig) * np.outer(p_sigma, p_sigma))
            C_new += cmu * rank_mu
            C = C_new
            
            C += 1e-12 * np.eye(n)
            try:
                C_cholesky = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_cholesky = eigvecs @ np.diag(np.sqrt(eigvals))
        
        return best_x, best_f
# EVOLVE-BLOCK-END
