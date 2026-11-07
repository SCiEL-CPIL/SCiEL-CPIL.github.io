import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, leave 10s margin
        
        # CMA-ES Parameters
        lambda_pop = 10 
        mu = max(1, lambda_pop // 2)
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # CMA-ES Strategy Parameters
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # Initial Mean and Step Size
        mean = np.zeros(d)
        sigma = 2.0  # Start with a larger step to explore the box [-5, 5]
        
        # State variables
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        C = np.eye(d)
        
        best_x = mean
        best_f = func(best_x)
        budget_used = 1
        
        while budget_used < budget:
            if time.time() > deadline:
                break
                
            # Sample lambda points
            z = np.random.randn(lambda_pop, d)
            
            # Transform to y
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(d)
                L = np.eye(d)
            
            Y = (z @ L.T) * sigma
            X = mean[None, :] + Y
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            budget_used += lambda_pop
            
            # Sort
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            Y = Y[idx]
            
            # Update Best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
                
            # Update Mean
            new_mean = np.sum(weights[:, None] * X[:mu], axis=0)
            
            # Update Covariance
            y_w = (new_mean - mean) / sigma
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
            
            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            # Heaviside function
            # chi_N approximation
            chi_N = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
            # Norm of p_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Threshold
            threshold = (1.4 + 2 / (d + 1)) * chi_N
            # We use a simplified check: if norm_p_sigma is large, h_sigma = 0
            # Standard check involves the expected norm, which is chi_N * sqrt(1 - (1-c_sigma)^...)
            # For simplicity and speed, we use:
            h_sigma = 1.0 if norm_p_sigma < 1.5 * chi_N else 0.0
            
            # Update C
            rank1 = np.outer(p_c, p_c)
            if h_sigma == 0.0:
                rank1 += c_sigma * (2 - c_sigma) * C
            
            Y_mu = Y[:mu]
            rank_mu = np.sum(weights[:, None, None] * Y_mu[:, :, None] * Y_mu[:, None, :], axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rank_mu
            C = (C + C.T) / 2
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_N - 1))
            
            # Update mean
            mean = new_mean

        return best_x, best_f
