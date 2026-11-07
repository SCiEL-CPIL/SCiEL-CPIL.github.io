import numpy as np

class Optimizer:
    """CMA-ES optimizer for BBOB f5 (Linear Slope)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = int(4 + 3 * np.log(d))
        
        # Initial parameters
        x = np.zeros(d) # Start at center, minimum is in [-4,4]
        sigma = 2.0
        C = np.eye(d)
        C_inv = np.eye(d)
        
        # Weights
        mu = lam // 2
        w = np.log(lam / 2 + 1) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Adaptation parameters
        cc = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        cs = (mu_eff + 2) / (d + mu_eff + 5)
        c1 = 2 / ((d + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + cs
        # Initial step size
        # For linear slope, the minimum is not at 0 necessarily, but close.
        # A moderate sigma is good.
        
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Track best solution
        x_best = x.copy()
        f_best = func(x_best)
        
        calls_used = 1
        
        # Precompute sqrt(C) for sampling
        # We will update C every generation
        
        for _ in range(budget - 1):
            # 1. Sample offspring
            Y = np.zeros((lam, d))
            for i in range(lam):
                # Sample from N(0, C)
                # Use Cholesky decomposition
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to identity if C is not positive definite
                    L = np.eye(d)
                z = np.random.randn(d)
                Y[i] = x + sigma * (L @ z)
            
            # Evaluate offspring
            # We have budget - calls_used remaining.
            # We need to evaluate lam offspring.
            # If lam > remaining budget, we reduce lam or stop.
            # For d=5, lam=9. Budget 500. 500/9 ~ 55 generations. This is fine.
            
            F = np.zeros(lam)
            for i in range(lam):
                if calls_used >= budget:
                    break
                # Handle boundaries: clip to [-5, 5]
                # Note: Clipping changes the distribution. 
                # A better approach is to check if out of bounds and penalize or resample.
                # Given the minimum is in [-4,4], clipping is usually safe and simple.
                y_i = np.clip(Y[i], -5, 5)
                F[i] = func(y_i)
                calls_used += 1
                if F[i] < f_best:
                    f_best = F[i]
                    x_best = y_i.copy()
            
            if calls_used >= budget:
                break
            
            # 2. Sort offspring
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]
            
            # 3. Update mean
            x_new = np.sum([w[i] * Y_sorted[i] for i in range(mu)], axis=0)
            
            # 4. Update covariance matrix
            # Standard deviation of the selected points
            Y_w = (Y_sorted[:mu] - x) / sigma
            # Covariance matrix update
            delta_sigma = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls_used - 1) / lam)) # Approximation
            # Actually, let's stick to standard CMA-ES update rules
            
            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * C_inv @ (x_new - x) / sigma
            
            # Update pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls_used) / lam)) < (1.4 + 2 / (d + 1)) * np.sqrt((mu_eff + 1) / (d + 1)))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x_new - x) / sigma
            
            # Update C
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (hsig - 1 + cc) * cc * C) 
            C += cmu * np.sum([w[i] * np.outer(Y_w[i], Y_w[i]) for i in range(mu)], axis=0)
            
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(2 * d) - 1))
            
            # Ensure C is positive definite
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update x
            x = x_new
            
            # Update C_inv for next iteration (approximation or recompute)
            # Recomputing inverse every step is expensive but d=5 is small.
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_inv = np.eye(d)
                C = np.eye(d) # Reset if singular

        return x_best, f_best
