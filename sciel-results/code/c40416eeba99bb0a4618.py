import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        N = 4 + int(3 * np.log(n))
        if N < 2:
            N = 2
            
        # Weights
        w = np.log((N + 1) / 2) - np.log(np.arange(1, N + 1))
        w /= np.sum(w)
        
        # Effective population size
        mu_eff = (np.sum(w) ** 2) / np.sum(w ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Initialization
        # Start at 0,0,0,0,0 which is inside [-4,4]^5
        x = np.zeros(n)
        sigma = 1.0
        C = np.eye(n)
        C_inv_sqrt = np.eye(n) # Actually we need Cholesky or sqrtm
        # Use Cholesky for sampling: x = mean + sigma * L @ z
        # C = L L^T
        L = np.eye(n)
        
        # CSA variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Initial evaluation
        best_x = x.copy()
        best_f = func(best_x)
        calls = 1
        
        # Time guard
        start_time = time.time()
        time_limit = 290.0 # Safety margin before 300s/330s limit
        
        # Pre-compute Cholesky of C
        # In CMA-ES, we maintain C and its Cholesky decomposition L such that C = L L^T
        # Sampling: x_i = x + sigma * L @ z_i
        
        for _ in range(budget - 1):
            if time.time() - start_time > time_limit:
                break
            
            # Generate N points
            Z = np.random.randn(N, n)
            # Transform Z using L: x_i = x + sigma * L @ Z_i.T
            # Since C = L L^T, we sample as x + sigma * L @ z
            # Note: Standard CMA-ES uses C^{1/2}. L is the lower triangular Cholesky factor.
            # X = x + sigma * (L @ Z.T).T
            X = x + sigma * (L @ Z.T).T
            
            # Clip to boundaries
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(xi) for xi in X])
            calls += N
            
            # Sort
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()
                
            # Weights for the best mu individuals
            mu = int(N / 2)
            w_mu = w[:mu]
            
            # Update mean
            x_new = x + sigma * (w_mu @ X_sorted[:mu])
            
            # Update step size (CSA)
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^{-1/2} * (x_new - x) / sigma
            # C^{-1/2} is the inverse Cholesky? No, it's the matrix square root inverse.
            # We can approximate or maintain it.
            # Actually, p_sigma update uses C^{-1/2}.
            # Let's maintain C_sqrt_inv or just use the fact that L is Cholesky of C.
            # C^{-1/2} is not simply L^{-1}.
            # However, for simplicity and speed in this small dimension, we can compute it or use a simplified update.
            # Standard implementation:
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (x_new - x) / sigma)
            
            # To avoid expensive matrix square root inverses every step, we can use the property that
            # if we have L, we can solve L y = b. But C^{-1/2} is different.
            # Let's stick to a robust implementation.
            # We can compute C_inv_sqrt using eigenvalue decomposition or similar, but that's O(n^3).
            # For n=5, it's fast.
            
            # Alternatively, we can skip the exact CSA and use a simpler sigma adaptation if it's stable enough.
            # But let's try to implement standard CMA-ES correctly.
            
            # Compute C_inv_sqrt
            # C = L L^T. C^{-1/2} is the matrix M such that M C M^T = I? No.
            # C^{1/2} is the symmetric square root.
            # Let's use np.linalg.eigh
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            d_norm = np.linalg.norm(C_inv_sqrt @ (x_new - x) / sigma)
            h_sigma = 1.0 if d_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // N + 1))) < (1 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1)) else 0.0
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new - x) / sigma
            
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(n) - 1)) # Wait, c_sigma in exp is different?
            # Standard: sigma *= exp(c_sigma * (||p_sigma|| / chi_n - 1))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update C
            # Rank-one update
            y = (x_new - x) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            C = (1 + c_1 * (2 - c_1) - c_mu) * C + c_1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            Y = (X_sorted[:mu] - x) / sigma
            C += c_mu * (w_mu @ (Y.T @ Y))
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            
            # Update Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not SPD, add small identity
                C += 1e-4 * np.eye(n)
                L = np.linalg.cholesky(C)
                
            x = x_new

        return best_x, best_f
