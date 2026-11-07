import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None
        self.deadline = 280.0  # 20s margin

    def __call__(self, func):
        self.start_time = time.time()
        n = self.dim
        budget = self.budget
        
        # Multi-start: 5 restarts
        n_restarts = 5
        budget_per_restart = max(20, budget // n_restarts)
        
        best_x = None
        best_f = float('inf')
        
        for r in range(n_restarts):
            if time.time() - self.start_time > self.deadline:
                break
                
            # Initialize
            # Start near center or random? Random is better for multimodal.
            # But f22 peaks are spread out.
            x_mean = np.random.uniform(-4.0, 4.0, n)
            sigma = 1.0
            C = np.eye(n)
            
            # Parameters
            lambda_ = 4 + int(3 * np.log(n))
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 1.5) / (n + mu_eff + 5)
            d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
            c_c = (4.0 + mu_eff / lambda_) / (n + 4.0 + mu_eff / lambda_)
            c_m = (mu_eff + 2.0) / (n + mu_eff + 5.0)
            c_1 = 2.0 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(c_1 / 2.0, 2.0 * (mu_eff - 2.0 + 2.0 / mu_eff) / ((n + 2.0)**2 + mu_eff))
            
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            # Evaluate initial mean
            f_mean = func(x_mean)
            if f_mean < best_f:
                best_f = f_mean
                best_x = x_mean.copy()
                
            # Main Loop
            for k in range(budget_per_restart - 1):
                if time.time() - self.start_time > self.deadline:
                    return best_x, best_f
                    
                # Sample
                ys = np.random.multivariate_normal(np.zeros(n), C, size=lambda_)
                xs = x_mean + sigma * ys
                xs = np.clip(xs, -5.0, 5.0)
                
                # Evaluate
                fs = np.array([func(x) for x in xs])
                
                # Update Best
                idx_min = np.argmin(fs)
                if fs[idx_min] < best_f:
                    best_f = fs[idx_min]
                    best_x = xs[idx_min].copy()
                
                # Sort
                order = np.argsort(fs)
                xs_sorted = xs[order]
                
                # Update Mean
                x_new_mean = np.sum(weights[:, np.newaxis] * xs_sorted[:mu], axis=0)
                
                # Update Sigma
                y_sigma = (x_new_mean - x_mean) / sigma
                p_sigma = (1.0 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * y_sigma
                norm_p_sigma = np.linalg.norm(p_sigma)
                chi_n = np.sqrt(n) * (1.0 - 1.0/(4.0*n) + 1.0/(21.0*n**2))
                # h_sigma
                # E[||N(0, C)||] approximation is chi_n if C=I, but C changes.
                # Standard heuristic:
                h_sigma = (norm_p_sigma / np.sqrt(1.0 - (1.0 - c_sigma)**(2 * (k + 1)))) < (1.4 + 2.0 / (n + 1.0)) * chi_n
                
                sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / np.sqrt(1.0 - (1.0 - c_sigma)**(2 * (k + 1))) - chi_n))
                
                # Update Covariance
                p_c = (1.0 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2.0 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma
                
                C = (1.0 - c_1 - c_mu + h_sigma * c_c * (2.0 - c_c)) * C
                C += c_1 * (np.outer(y_sigma, y_sigma) + (1.0 - h_sigma) * c_c * (2.0 - c_c) * np.outer(p_c, p_c))
                
                # Rank-mu
                y_i = (xs_sorted[:mu] - x_mean) / sigma
                for i in range(mu):
                    C += c_mu * weights[i] * np.outer(y_i[i], y_i[i])
                    
                x_mean = x_new_mean
                
        return best_x, best_f
