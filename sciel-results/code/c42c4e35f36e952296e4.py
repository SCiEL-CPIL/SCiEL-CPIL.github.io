import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # For n=5, lambda=11 is a good choice
        lam = 11 
        mu = lam // 2
        
        # Weights for mean update
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / (weights ** 2).sum()
        
        # Strategy parameters
        cs = (lam/2 + 2) / (n + lam/2 + 5)
        ds = 1 + 2 * max(0, np.sqrt((lam/2 - 1) / (n + 1)) - 1) + cs
        # cmu is not directly used in this simplified update, but c_mu is
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Time guard
        start_time = time.time()
        time_limit = 295.0
        
        # Initialization
        # Start at the center of the box
        mean = np.zeros(n)
        sigma = 2.0
        
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        best_x = mean.copy()
        best_f = func(best_x)
        evals = 1
        
        while evals < budget:
            if time.time() - start_time > time_limit:
                break
                
            # If sigma is too small, we are likely converged or stuck
            if sigma < 1e-12:
                break
                
            # Sample candidates
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is singular, reset to identity
                C = np.eye(n)
                L = np.eye(n)
                
            z_samples = np.random.randn(lam, n)
            # Transform to match covariance C
            # x = mean + sigma * L @ z
            x_samples = mean + sigma * (L @ z_samples.T).T
            
            f_samples = np.full(lam, np.inf)
            
            # Evaluate candidates
            for i in range(lam):
                if time.time() - start_time > time_limit:
                    break
                x_i = np.clip(x_samples[i], -5, 5)
                f_i = func(x_i)
                f_samples[i] = f_i
                evals += 1
                
                # Track global best
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
            
            # If we ran out of time or budget mid-generation, break
            if np.any(np.isinf(f_samples)):
                break
            
            # Sort samples by fitness
            order = np.argsort(f_samples)
            x_samples = x_samples[order]
            f_samples = f_samples[order]
            
            # Update mean
            new_mean = np.sum(weights[:, None] * x_samples[:mu], axis=0)
            
            # Update evolution path ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (L.T @ (new_mean - mean))
            
            # Check for sigma adaptation
            # hsig condition
            norm_ps = np.linalg.norm(ps)
            # Threshold
            # Using a simplified check
            hsig = 1.0 if norm_ps / np.sqrt(1 - (1 - cs)**(2 * (evals // lam + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt(n) else 0.0
            
            if hsig:
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / np.sqrt(n) - 1))
            
            # Update evolution path pc
            pc = (1 - c_c) * pc + hsig * c_c / sigma * (new_mean - mean)
            
            # Update Covariance Matrix
            delta_C = np.zeros((n, n))
            for i in range(mu):
                yi = (x_samples[i] - mean) / sigma
                delta_C += weights[i] * np.outer(yi, yi)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - hsig) * c_c * (2 - c_c) * C) + c_mu * delta_C
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-10:
                eigvals = np.clip(eigvals, 1e-10, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            mean = new_mean

        return best_x, best_f
