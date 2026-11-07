"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size lambda. For dim=5, lambda=10 is standard.
        lam = 10
        mu = lam // 2
        
        # Weights for recombination
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        
        # Mu_eff: effective population size
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial step size
        sigma = 1.0
        
        # Initial mean: center of the box
        mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution path for covariance
        p_c = np.zeros(dim)
        # Evolution path for sigma
        p_sigma = np.zeros(dim)
        
        # Initial evaluation
        x0 = np.clip(mean, -5, 5)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        
        call_count = 1
        
        while call_count < budget:
            # Check deadline
            if time.time() > deadline:
                break
                
            # Compute sqrt(C)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 0, None)
            sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals))
            
            # Generate population
            X = np.zeros((lam, dim))
            F = np.zeros(lam)
            
            for i in range(lam):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                z = np.random.randn(dim)
                x_i = np.clip(mean + sigma * (sqrt_C @ z), -5, 5)
                X[i] = x_i
                F[i] = func(x_i)
                call_count += 1
                
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x_i.copy()
            
            # If we didn't complete a full generation, break
            if call_count >= budget or time.time() > deadline:
                break
                
            # Sort by fitness
            idx = np.argsort(F)
            F_sorted = F[idx]
            X_sorted = X[idx]
            
            # Update mean
            mean_old = mean.copy()
            mean_new = np.sum(w[:, None] * X_sorted[:mu], axis=0)
            
            # Update evolution paths
            y = (mean_new - mean_old) / sigma
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (sqrt_C.T @ y)
            
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) / np.sqrt(np.trace(C)) < (1 + 1.4 / dim) * np.sqrt((mu_eff + 1) / (dim + 1)) else 0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            
            # Update Covariance
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                z_i = (X_sorted[i] - mean_old) / sigma
                rank_mu += w[i] * np.outer(z_i, z_i)
                
            C = (1 + c_1 * h_sigma * (2 - c_c) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(np.trace(C)) - 1))
            
            mean = mean_new
            
            # Optional: Stopping criteria
            # If sigma is very small or variance is very small, we might be stuck
            if sigma < 1e-12:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
