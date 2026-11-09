import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = 280.0
        n = self.dim
        
        # 1. Global Exploration
        # Use 20% of budget or fixed 50 calls for random search
        n_global = 50
        if n_global > self.budget:
            n_global = self.budget // 2
            
        rng = np.random.default_rng(0)
        x_candidates = rng.uniform(-4, 4, (n_global, n))
        f_candidates = np.array([func(x) for x in x_candidates])
        
        idx_best = np.argmin(f_candidates)
        x_m = x_candidates[idx_best].copy()
        f_best = f_candidates[idx_best]
        
        calls_made = n_global
        
        # 2. Local Refinement with CMA-ES
        lambda_pop = 4 + int(3 * np.log(n))
        mu = lambda_pop // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        sigma = 1.0 # Start smaller since we have a good global start
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        cs = (1 + c_sigma + c_c) * 0.5
        gen = 0
        
        while calls_made < self.budget:
            if time.time() - t_start > deadline:
                break
                
            # Generate population
            Y = rng.standard_normal((lambda_pop, n))
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity or add small epsilon
                C = np.eye(n)
                L = np.eye(n)
                
            X = x_m + sigma * (Y @ L.T)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls_made += lambda_pop
            gen += 1
            
            # Update best
            idx_f_best = np.argmin(F)
            if F[idx_f_best] < f_best:
                f_best = F[idx_f_best]
                x_best = X[idx_f_best].copy()
            else:
                x_best = x_m # Keep previous best if no improvement in this gen? 
                             # Actually, x_m is the weighted mean, which might be better.
                             # But we track f_best from all evaluations.
            
            # Sort
            idx_sorted = np.argsort(F)
            X_sorted = X[idx_sorted]
            
            # Update mean
            x_new = np.sum(w[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # Update evolution paths
            L_inv = np.linalg.inv(L)
            y = (x_new - x_m) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L_inv.T @ y)
            
            # hsig check
            norm_ps = np.linalg.norm(ps)
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(25*n**2))
            hsig = norm_ps / (np.sqrt(1 - (1 - c_sigma)**(2 * gen)) * chi_n) < (1 + 2 / (n + 1))
            
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            
            # Update Covariance
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            diff = X_sorted[:mu] - x_m
            rank_mu = np.sum(w[:, np.newaxis] * np.outer(diff, diff), axis=0) / sigma**2
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + delta_hsig * C) + c_mu * rank_mu
            
            # Regularize C to ensure positive definiteness
            C = (C + C.T) / 2
            # Add small diagonal if needed
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += (1e-12 - np.min(eigvals)) * np.eye(n)
                
            # Update sigma
            sigma = sigma * np.exp(c_sigma / cs * (norm_ps / chi_n - 1))
            
            x_m = x_new
            x_m = np.clip(x_m, -5, 5)
            
        return x_best, f_best
