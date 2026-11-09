"""Rewritten optimizer using CMA-ES for efficient multimodal optimization."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB-style f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(dim))
        x_mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        
        # Eigenvalue decomposition
        # B: eigenvectors, D: eigenvalues (we store sqrt(D) as D_sqrt)
        B = np.eye(dim)
        D_sqrt = np.ones(dim)
        
        # Adaptation parameters
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        budget_used = 1
        
        while budget_used + lam <= budget:
            # Generate population
            xs = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                x = x_mean + sigma * (B @ (D_sqrt * z))
                x = np.clip(x, -5, 5)
                xs.append(x)
            
            # Evaluate
            fs = [func(x) for x in xs]
            budget_used += lam
            
            # Sort
            idx = np.argsort(fs)
            xs_sorted = [xs[i] for i in idx]
            
            # Update best
            if fs[idx[0]] < best_f:
                best_f = fs[idx[0]]
                best_x = xs_sorted[0].copy()
            
            # Weighted mean
            x_new = sum(w[i] * xs_sorted[i] for i in range(mu))
            
            # Update p_sigma
            # C^{-1/2} = B D^{-1} B^T
            diff = x_new - x_mean
            # z_diff = C^{-1/2} diff = B @ (D^{-1} @ (B.T @ diff))
            z_diff = B @ ((B.T @ diff) / D_sqrt)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_diff / sigma
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * z_diff
            
            # Update C
            delta_h = (1 - np.exp(-c_c)) * np.linalg.norm(p_c) / chi_n + 2 - np.exp(-c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            for i in range(mu):
                diff_i = xs_sorted[i] - x_new
                z_i = B @ ((B.T @ diff_i) / D_sqrt)
                C += c_mu * w[i] * np.outer(z_i, z_i)
            
            C = (C + C.T) / 2
            
            # Refactorize
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            B = eigvecs
            D_sqrt = np.sqrt(eigvals)
            
            x_mean = x_new
            
        return best_x, best_f
# EVOLVE-BLOCK-END
