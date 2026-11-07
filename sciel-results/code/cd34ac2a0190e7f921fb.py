"""Optimized program using CMA-ES for BBOB f9 (Rotated Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        
        # Helper to update C_sqrt and C_inv
        def refresh_cmat(C):
            try:
                L = np.linalg.cholesky(C)
                C_inv = np.linalg.inv(L).T @ np.linalg.inv(L)
                return L, C_inv
            except np.linalg.LinAlgError:
                C_reg = C + 1e-10 * np.eye(dim)
                L = np.linalg.cholesky(C_reg)
                C_inv = np.linalg.inv(L).T @ np.linalg.inv(L)
                return L, C_inv

        C_sqrt, C_inv = refresh_cmat(C)
        
        # Initial evaluation
        best_x = x_mean.copy()
        best_f = func(best_x)
        eval_count = 1
        
        # Main Loop
        while eval_count + lam <= budget:
            if time.time() > deadline:
                break
                
            # Generate Population
            Y = []
            F = []
            for i in range(lam):
                z = np.random.randn(dim)
                y = x_mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                Y.append(y)
                F.append(fy)
                eval_count += 1
                
            # Sort by fitness
            idx = np.argsort(F)
            Y = [Y[i] for i in idx]
            F = [F[i] for i in idx]
            
            # Update Best
            if F[0] < best_f:
                best_f = F[0]
                best_x = Y[0].copy()
                
            # Recombination
            x_new = sum(w[i] * Y[i] for i in range(mu))
            
            # Update p_sigma
            y_sigma = (C_sqrt @ (C_inv @ (x_new - x_mean))) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            # Update sigma
            # Standard adaptation
            # t_sigma is the number of generations since start
            # We track generations implicitly
            # Simplified:
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Approximate expected norm
            # Using a simplified version of the Hansen formula
            # sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * (eval_count - (eval_count - lam)))) - np.sqrt(dim)))
            # This term (eval_count - (eval_count - lam)) is just lam.
            # So:
            denom = np.sqrt(1 - (1 - c_sigma)**(2 * lam))
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / denom - np.sqrt(dim)))
            sigma = np.clip(sigma, 1e-10, 1e10)
            
            # Update p_c
            h_sigma = norm_p_sigma / denom < (1 + 2 / (dim + 1)) * np.sqrt((mu_eff + 2) / (dim + mu_eff + 5))
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv @ (x_new - x_mean)) / sigma
            
            # Update C
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_mu_matrix = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (Y[i] - x_mean) / sigma
                rank_mu_matrix += w[i] * np.outer(y_i, y_i)
                
            C = (1 + c_1 * c_c + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * rank_mu_matrix
            C = (C + C.T) / 2
            
            # Update mean
            x_mean = x_new
            
            # Refresh C_sqrt, C_inv
            C_sqrt, C_inv = refresh_cmat(C)
            
        # Use remaining budget for local search if needed
        # For CMA-ES, the best point is usually very good.
        
        return best_x, best_f

# EVOLVE-BLOCK-END
