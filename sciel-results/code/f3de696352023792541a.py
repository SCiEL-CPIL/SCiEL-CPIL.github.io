import numpy as np
import time

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
    Uses a population of lambda candidates per generation, adapting the
    covariance matrix to the problem's landscape geometry.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, stop at 320s to be safe
        deadline = start_time + 320.0
        
        rng = np.random.default_rng(42)
        n = self.dim
        
        # CMA-ES parameters
        # lambda is typically 4 + floor(3 * log(n))
        lam = 4 + int(3 * np.log(n))
        # mu is the number of best individuals used for update, typically lambda // 2
        mu = lam // 2
        
        # Weights for the best mu individuals
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Step size adaptation parameters
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initial state
        # Start near center, but with some spread
        x_mean = np.zeros(n)
        sigma = 2.0  # Initial step size
        C = np.eye(n)
        C_sqrt_inv = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Track best
        best_fx = np.inf
        best_x = np.zeros(n)
        
        calls_used = 0
        
        while calls_used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate population
            ys = np.zeros((lam, n))
            fs = np.zeros(lam)
            
            for i in range(lam):
                z = rng.standard_normal(n)
                y = x_mean + sigma * (C_sqrt_inv @ z)
                y = np.clip(y, -5, 5)
                ys[i] = y
                fs[i] = func(y)
                calls_used += 1
                
                if fs[i] < best_fx:
                    best_fx = fs[i]
                    best_x = y.copy()
                    
                # If we hit the deadline during population generation, we can't easily
                # finish the generation. We just break out.
                if time.time() > deadline:
                    break
            
            if calls_used >= self.budget or time.time() > deadline:
                break
                
            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]
            
            # Update mean
            x_new = np.sum(w[:, None] * ys[:mu], axis=0)
            
            # Update covariance matrix
            y_inv = np.linalg.solve(C_sqrt_inv, (x_new - x_mean) / sigma)
            
            # Update path accumulators
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * y_inv
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**((calls_used / (lam * cs) + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1)))
            
            p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * y_inv
            
            # Rank-one and rank-mu updates
            C = (1 + c1 * hsig * (hsig - 1 + cc) + cmu) * C \
                - c1 * np.outer(p_c, p_c) \
                - cmu / sigma**2 * np.sum(w[:, None, None] * np.outer(ys[:mu] - x_mean, ys[:mu] - x_mean), axis=0)
                
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            
            # Update inverse square root
            try:
                C_sqrt_inv = np.linalg.inv(np.linalg.cholesky(C))
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset C to identity and recompute
                C = np.eye(n)
                C_sqrt_inv = np.eye(n)
                
            # Update sigma
            sigma = sigma * np.exp(cs / (n + 4) * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update mean
            x_mean = x_new
            
            # Optional: restart if sigma becomes too small or too large
            if sigma < 1e-10 or sigma > 1e10:
                # Restart with new random point
                x_mean = rng.uniform(-4, 4, n)
                sigma = 2.0
                C = np.eye(n)
                C_sqrt_inv = np.eye(n)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                
                # Evaluate the new mean to ensure we have a valid point
                fx_new = func(x_mean)
                calls_used += 1
                if fx_new < best_fx:
                    best_fx = fx_new
                    best_x = x_mean.copy()

        return best_x, best_fx
