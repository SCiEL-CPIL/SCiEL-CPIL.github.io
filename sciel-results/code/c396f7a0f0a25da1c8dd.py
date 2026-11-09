import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin for 300s limit
        
        # CMA-ES Parameters
        dim = self.dim
        lambda_ = 4
        mu = lambda_ // 2
        
        # Weights
        w = np.zeros(mu)
        w[:mu] = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1 / np.sum(w**2)
        
        # Strategy Parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initialization
        x_mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        B, D = np.linalg.eigh(C)
        inv_sqrt_C = np.diag(1.0 / np.sqrt(D))
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_used = 1
        
        while calls_used < self.budget:
            if time.time() > deadline:
                break
                
            # Sample population
            ys = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = np.random.randn(dim)
                y = x_mean + sigma * (B @ (inv_sqrt_C @ z))
                ys[i] = np.clip(y, -5, 5)
            
            # Evaluate
            fs = np.zeros(lambda_)
            for i in range(lambda_):
                fs[i] = func(ys[i])
                calls_used += 1
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = ys[i].copy()
            
            # Sort
            order = np.argsort(fs)
            y_sorted = ys[order]
            f_sorted = fs[order]
            
            # Weighted mean
            y_weighted = np.sum(w[:, None] * y_sorted[:mu], axis=0)
            
            # Update paths
            y_diff = y_weighted - x_mean
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * y_diff
            
            # h_sigma check (simplified version for stability)
            # Standard check involves trace(C) and norms
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Approximation for h_sigma to avoid complex trace calculations in tight loop if possible, 
            # but standard CMA-ES requires it.
            # Let's use the standard formula.
            # The term (1 - c_sigma)^(2 * gen) is approximated. 
            # For simplicity and speed, we can often skip h_sigma or use a simpler heuristic, 
            # but let's try to include it properly or simplify.
            # Given the small budget, skipping h_sigma might lead to sigma collapse too fast or slow.
            # Let's use a simplified h_sigma = 1.0 if norm_p_sigma is not too large.
            # Actually, for dim=5, we can afford the calculation.
            
            # Recalculate trace(C) is expensive. We have D.
            trace_C = np.sum(D)
            # The condition for h_sigma
            # (norm(p_sigma) / sqrt(1 - (1-c_sigma)^(2*gen))) < (1.4 + 2/(dim+1)) * sqrt(mu_eff) * sqrt(trace_C)
            # We don't track gen exactly in the exponent easily without a counter, but we can use calls_used.
            # Let's just set h_sigma = 1.0 for simplicity, as the adaptation of C is the main driver.
            # If we set h_sigma=1, p_c updates fully.
            h_sigma = 1.0 
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * y_diff
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_1 = np.outer(p_c, p_c)
            
            # Rank-mu update
            # rank_mu = sum(w_i * outer(y_i - x_mean, y_i - x_mean)) / sigma^2
            # Efficient calculation:
            # Y_diff = (y_sorted[:mu] - x_mean[None, :]) / sigma
            # rank_mu = (w[:, None] * Y_diff.T) @ Y_diff
            Y_diff = (y_sorted[:mu] - x_mean[None, :]) / sigma
            rank_mu = (w[:, None] * Y_diff.T) @ Y_diff
            
            C = (1 + c_1 * (delta_h + 1 - np.trace(C)) + c_mu * (np.trace(rank_mu) - np.trace(C))) * C + c_1 * rank_1 + c_mu * rank_mu
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Eigen decomposition
            B, D = np.linalg.eigh(C)
            # Clip eigenvalues
            D = np.clip(D, 1e-12, None)
            inv_sqrt_C = np.diag(1.0 / np.sqrt(D))
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / np.sqrt(mu_eff) - 1))
            
            # Update mean
            x_mean = y_weighted
            
            # Check for convergence/stall
            if sigma < 1e-8:
                break

        return best_x, best_f
