"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer with adaptive step size for bent cigar functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Population size
        lam = max(4, int(4 + 3 * np.log(n)))
        
        # Initial mean and sigma
        x_mean = np.zeros(n)
        sigma = 1.5
        
        # CMA-ES parameters
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = int(np.sum(weights[:mu]) ** 2 / np.sum(weights ** 2))
        
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 2 / mueff) / ((n + 2) ** 2 + mueff))
        
        # Evolution path
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Covariance matrix
        C = np.eye(n)
        sqrtC = np.eye(n)
        inv_sqrtC = np.eye(n)
        
        # Precompute for efficient sampling
        # We'll use a simple Cholesky-like update or just use the matrix directly
        # For small n (n=5), we can just do matrix operations directly
        
        def cholesky(A):
            L = np.linalg.cholesky(A + 1e-10 * np.eye(n))
            return L
        
        def inv_cholesky(A):
            L = cholesky(A)
            return np.linalg.inv(L)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        # Generate initial population
        def generate_population():
            nonlocal x_mean, sigma, p_sigma, p_c, C, sqrtC, inv_sqrtC, calls, best_x, best_f
            
            # Update sqrtC and inv_sqrtC
            try:
                C = 0.5 * C + 0.5 * C  # ensure symmetry
                L = cholesky(C)
                sqrtC = L
                inv_sqrtC = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # Fallback to identity if Cholesky fails
                C = np.eye(n)
                sqrtC = np.eye(n)
                inv_sqrtC = np.eye(n)
            
            # Generate lambda samples
            y_samples = rng.standard_normal((lam, n))
            x_samples = x_mean + sigma * (y_samples @ inv_sqrtC.T)
            
            # Evaluate
            f_values = np.zeros(lam)
            for i in range(lam):
                x_i = np.clip(x_samples[i], -5, 5)
                f_values[i] = func(x_i)
                calls += 1
                if f_values[i] < best_f:
                    best_f = f_values[i]
                    best_x = x_i.copy()
            
            # Sort
            idx = np.argsort(f_values)
            f_sorted = f_values[idx]
            x_sorted = x_samples[idx]
            
            # Update mean
            x_old = x_mean.copy()
            x_mean_new = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            
            # Update evolution paths
            # p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (inv_sqrtC.T @ (x_mean_new - x_old) / sigma)
            
            # h_sigma
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            h_sigma = np.min(1, (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls + 1))) / chi_n)
            
            # p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (x_mean_new - x_old) / sigma
            
            # Update covariance matrix
            delta = (h_sigma * (2 - c_c) - 1) * np.outer(p_c, p_c)
            C = (1 + c_1 + c_mu - c_1 - c_c) * C \
                + c_1 * (np.outer(x_mean_new - x_old, x_mean_new - x_old) / sigma ** 2 - C) \
                + c_mu * np.sum(weights[:, None, None] * np.einsum('i,ij,ik->ijk', x_sorted[:mu] - x_old, x_sorted[:mu] - x_old, x_sorted[:mu] - x_old), axis=0) / sigma ** 2 \
                + delta
            
            # Ensure symmetry and positive semi-definiteness
            C = (C + C.T) / 2
            # Regularize if needed
            try:
                np.linalg.cholesky(C + 1e-10 * np.eye(n))
            except np.linalg.LinAlgError:
                C = C + 1e-6 * np.eye(n)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            return x_mean_new, sigma
        
        # Main loop
        while calls < budget:
            if budget - calls < lam:
                # Only do single steps
                y = rng.standard_normal(n)
                x_new = np.clip(x_mean + sigma * (inv_sqrtC.T @ y), -5, 5)
                f_new = func(x_new)
                calls += 1
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                    x_mean = x_new
                else:
                    # Small adjustment
                    pass
            else:
                x_mean, sigma = generate_population()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
