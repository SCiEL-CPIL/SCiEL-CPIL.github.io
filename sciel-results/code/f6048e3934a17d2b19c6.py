import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin
        
        d = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # For small dimensions, standard parameters work well
        lambda_ = 4 + int(3 * np.log(d))  # Population size
        # Ensure lambda is at least 2
        lambda_ = max(lambda_ + 1, 2) # Usually lambda is even or specific, let's stick to standard 4+3log(d) but ensure valid
        # Standard CMA-ES lambda is often just 4 for d=5? 
        # Formula: mu = lambda // 2. 
        # Let's use a fixed lambda=4 for d=5 to keep it simple and fast, or standard.
        # Standard: lambda = 4 + int(3*log(5)) = 4 + 4 = 8. mu = 4.
        # With budget 500, 500/8 = 62 generations.
        # Let's try lambda=4 to get more generations (125).
        # CMA-ES is sensitive to lambda. For d=5, lambda=4 is small but often effective for quick convergence in low dims.
        # Let's use lambda = 4 + int(3 * np.log(d)) which is 8 for d=5.
        lambda_pop = 4 + int(3 * np.log(d))
        mu = lambda_pop // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # Initial state
        # Start at origin? Or random?
        # The problem says minimizer is in [-4, 4]^5. 
        # Starting at 0 is a reasonable heuristic for symmetric-ish distributions, 
        # but with random shifts, 0 might be far.
        # However, CMA-ES adapts. 
        # Let's start at 0.0 vector.
        x = np.zeros(d)
        
        # Initial sigma
        # Box is [-5, 5]. 
        # A reasonable initial step is 2.0 or 1.0.
        # Rosenbrock valley is narrow. 
        # Let's try sigma = 2.0.
        sigma = 2.0
        
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        best_x = x.copy()
        best_f = func(x)
        calls = 1
        
        # Pre-generate random numbers? No, we need to stop early if time runs out.
        # We will loop until budget is reached or time limit.
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Generate population
            Y = np.zeros((lambda_pop, d))
            for i in range(lambda_pop):
                Y[i] = x + sigma * C @ np.random.multivariate_normal(np.zeros(d), np.eye(d))
                # Clip to box? 
                # The task says func clips. 
                # However, CMA-ES works best if the distribution stays within bounds 
                # or if we handle reflections. 
                # Since func clips, we can just pass Y. 
                # But if Y is far outside, it wastes evals. 
                # With sigma=2 and x in [-4,4], Y is likely in [-8, 8]. 
                # Clipping to [-5,5] is fine.
            
            # Evaluate
            F = np.zeros(lambda_pop)
            for i in range(lambda_pop):
                F[i] = func(Y[i])
                calls += 1
                if time.time() > deadline:
                    # If we run out of time mid-generation, we can't finish the generation
                    # We should probably break out of the inner loop and outer loop
                    # But we've already evaluated some. 
                    # It's safer to break the whole process.
                    break
            
            if time.time() > deadline:
                break
            
            # Update best
            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = Y[idx_best].copy()
            
            # Sort population
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]
            
            # Update mean
            w = weights[:mu]
            x_new = np.sum([w[i] * Y_sorted[i] for i in range(mu)], axis=0)
            
            # Update covariance matrix
            # Standard CMA-ES update
            # D = C ** 0.5 (matrix square root)
            # Using eigendecomposition for stability
            eigval, eigvec = np.linalg.eigh(C)
            D = eigvec @ np.diag(np.sqrt(eigval))
            D_inv = eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * D_inv @ (x_new - x)
            
            # Update pc
            h_sig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls - d))) < (1 + 2 / (d + 1)) * chi_d
            # Note: calls here is cumulative, but CMA-ES theory uses generation count. 
            # Let's use generation count.
            gen = calls // lambda_pop
            h_sig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * gen)) < (1 + 2 / (d + 1)) * chi_d
            
            if h_sig:
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
            
            # Rank-mu update
            v = np.zeros((d, d))
            for i in range(mu):
                y_i = D_inv @ (Y_sorted[i] - x)
                v += w[i] * np.outer(y_i, y_i)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - h_sig) * c_c * (2 - c_c) * C) + c_mu * v
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Sometimes numerical issues arise, add small regularization
            C += 1e-10 * np.eye(d)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_d - 1))
            
            # Move to new mean
            x = x_new
            
            # Optional: If we get stuck or sigma becomes too small/large
            if sigma < 1e-12:
                break
            if sigma > 1e6:
                break

        return best_x, best_f
