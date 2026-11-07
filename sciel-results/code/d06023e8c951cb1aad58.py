import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1. / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1) - 1)) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim**2))
        
        x_mean = np.zeros(dim)
        sigma = 2.0
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = np.inf
        
        calls = 0
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Sample
            Y = np.array([x_mean + sigma * (B @ (D * np.random.standard_normal(dim))) for _ in range(lambda_)])
            
            # Evaluate
            F = np.array([func(y) for y in Y if calls < budget])
            calls += len(F)
            
            if len(F) < lambda_:
                break
                
            # Sort
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]
            
            # Update mean
            x_mean_new = np.sum(weights[:, np.newaxis] * Y[:mu], axis=0)
            
            # Update p_sigma
            # Inverse of B @ diag(D)
            # (B @ diag(D))^-1 = diag(1/D) @ B^-1
            z = np.linalg.solve(B, (x_mean_new - x_mean) / sigma / D)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * z
            
            # Update Covariance
            C = B @ np.diag(D**2) @ B.T
            h_sigma = min(1, (np.linalg.norm(p_sigma) / chi_n) ** 2 / (2 * dim) * (1 + 2 * (c_sigma / d_sigma - 1)))
            # h_sigma is usually 1 if norm > 1.4 * chi_n, else 0? 
            # Standard: h_sigma = 1 if ||p_sigma||/chi_n > 1.4 * sqrt((1-(1-c_sigma)^{2/gen}) / (1-(1-c_sigma)^{2/(gen*lambda)}))
            # Simplified:
            # Let's use the rank-mu update primarily, and rank-1 if condition met.
            # For robustness, let's just do rank-mu update which is stable.
            # C_new = (1 - c_c - c_c * mu_eff) * C + c_c * (outer(p_c, p_c) + 2 * c_sigma * C) + c_mu * sum(...)
            # c_mu = 2 * (mu_eff - 2 + 2/mu_eff) / ((dim + 2)^2 + mu_eff)
            c_mu = 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff)
            
            # Rank-1 part
            # We need to check if h_sigma is 1.
            # If we skip rank-1, convergence is slower but robust.
            # Let's include rank-1 with a simple check: if norm(p_sigma) > 1.4 * chi_n
            if np.linalg.norm(p_sigma) > 1.4 * chi_n:
                rank1 = np.outer(p_c, p_c)
                C_new = (1 - c_c - c_c * mu_eff) * C + c_c * (rank1 + 2 * c_sigma * C)
            else:
                C_new = (1 - c_c - c_c * mu_eff) * C
            
            # Rank-mu part
            Y_mu = (Y[:mu] - x_mean) / sigma
            rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * (Y_mu[:, np.newaxis, :] * Y_mu[:, np.newaxis, :].T), axis=0)
            C_new += c_mu * rank_mu
            
            # Eigen decomposition for B and D
            eigvals, eigvecs = np.linalg.eigh(C_new)
            # Ensure positive definite
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.sqrt(eigvals)
            B = eigvecs
            
            x_mean = x_mean_new
            
            # Track best
            current_best = F[0]
            if current_best < best_f:
                best_f = current_best
                best_x = Y[0].copy()
                
        return best_x, best_f
