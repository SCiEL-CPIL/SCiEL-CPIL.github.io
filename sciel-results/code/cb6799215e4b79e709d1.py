import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # lambda is the population size
        lam = 10 
        # k is the number of parents
        k = lam // 2 + 1
        # mu_w are the recombination weights
        weights = np.log(k + 1) - np.log(np.arange(1, k + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Initial mean
        x_mean = np.zeros(dim)
        # Initial step size
        sigma = 0.5
        
        # Initial covariance matrix (identity)
        C = np.eye(dim)
        C_inv = np.eye(dim)
        # Precompute C^{1/2} for sampling
        C_sqrt = np.eye(dim)
        
        # Evaluation history for best tracking
        best_x = x_mean.copy()
        best_fx = func(best_x)
        calls = 1
        
        # Path variables for step size and covariance adaptation
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + 4)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_mu = min(1 - c_c, 2 * (mu_eff - 2 + 1 / mu_eff) / (dim + 2))
        
        # Chi-square expectation correction
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # 1. Sample lambda points
            ys = []
            xs = []
            for i in range(lam):
                # Sample y from N(0, I)
                y = np.random.normal(0, 1, dim)
                # Transform to x = mean + sigma * C^{1/2} y
                x = x_mean + sigma * (C_sqrt @ y)
                # Clip to bounds
                x = np.clip(x, -5, 5)
                ys.append(y)
                xs.append(x)
                
            # 2. Evaluate
            f_vals = []
            for x in xs:
                f = func(x)
                f_vals.append(f)
                calls += 1
                if f < best_fx:
                    best_fx = f
                    best_x = x.copy()
                    
            # 3. Sort by fitness
            indices = np.argsort(f_vals)
            ys_sorted = [ys[i] for i in indices]
            xs_sorted = [xs[i] for i in indices]
            
            # 4. Update mean
            y_w = np.sum([weights[i] * ys_sorted[i] for i in range(k)], axis=0)
            x_mean += sigma * c_sigma * y_w
            x_mean = np.clip(x_mean, -5, 5)
            
            # 5. Update step size
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt @ y_w) / sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # 6. Update covariance matrix
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_sqrt @ y_w)
            
            # Rank-one update
            C += c_mu * (np.outer(y_w, y_w) + (1 - (1 - c_mu) * np.outer(p_c, p_c) / (dim * sigma**2)) * C)
            # Simplified rank-one update based on standard CMA-ES
            # C = (1 - c_mu) * C + c_mu * (np.outer(y_w, y_w) + (1 - c_mu) * np.eye(dim))
            # Actually, the standard update is:
            # C = (1 - c_mu) * C + c_mu * (np.outer(y_w, y_w) + (1 - c_mu) * np.eye(dim))
            # But we need to be careful with the p_c term.
            # Let's use the simpler rank-one update for stability and speed.
            C = (1 - c_mu) * C + c_mu * (np.outer(y_w, y_w) + (1 - c_mu) * np.eye(dim))
            
            # Symmetrize and ensure positive semi-definite
            C = (C + C.T) / 2
            # Add small regularization to prevent singularity
            C += 1e-10 * np.eye(dim)
            
            # Update C_inv and C_sqrt
            try:
                C_inv = np.linalg.inv(C)
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If singular, restart covariance
                C = np.eye(dim)
                C_inv = np.eye(dim)
                C_sqrt = np.eye(dim)
                sigma = 0.5
                x_mean = best_x.copy()
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)

        return best_x, best_fx
