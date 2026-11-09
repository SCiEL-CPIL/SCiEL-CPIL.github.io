import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 325.0  # Leave 5s margin for the 330s kill
        
        def eval_func(x):
            if self.calls >= self.budget:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            
            # Ensure x is finite and valid before passing to func
            if not np.all(np.isfinite(x)):
                # If x is invalid, return a large penalty or the best known value
                # But we should ideally not generate invalid x.
                # For safety, if this happens, treat it as a very bad point.
                self.calls += 1
                return 1e100
            
            self.calls += 1
            try:
                fx = func(x)
                if not np.isfinite(fx):
                    return 1e100
                return fx
            except Exception:
                return 1e100

        n = self.dim
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(n))
        # Initial step size
        sigma = 1.0
        # Initial mean (center of the box)
        m = np.zeros(n)
        # Initial covariance (identity)
        C = np.eye(n)
        # Inverse of C
        C_inv = np.eye(n)
        # Cholesky decomposition of C
        L = np.eye(n)
        
        # Weights
        mu = lambda_ // 2
        w = np.log(lambda_ / 2 + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Update parameters
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        ps = np.zeros(n)
        best_x = m.copy()
        best_fx = float('inf')
        
        try:
            while self.calls < self.budget and (time.time() - self.start_time) < self.time_limit:
                # Generate population
                pop = np.zeros((lambda_, n))
                fits = np.zeros(lambda_)
                
                for i in range(lambda_):
                    # Check time/budget before sampling
                    if self.calls >= self.budget or (time.time() - self.start_time) > self.time_limit:
                        break
                    
                    # Sample
                    z = np.random.randn(n)
                    x = m + sigma * (L @ z)
                    
                    # Clip to box [-5, 5] to ensure validity for some functions, 
                    # though the problem says func clips. But keeping x finite is crucial.
                    x = np.clip(x, -5.0, 5.0)
                    
                    pop[i] = x
                    # Evaluate
                    try:
                        fits[i] = eval_func(x)
                    except StopIteration:
                        break
                
                if self.calls >= self.budget or (time.time() - self.start_time) > self.time_limit:
                    break
                    
                # Check if we have enough valid evaluations to proceed
                # If all fits are 1e100 (invalid), we might want to reset or shrink sigma
                if np.all(fits == 1e100):
                    sigma *= 0.5
                    continue

                # Sort by fitness
                order = np.argsort(fits)
                sorted_fits = fits[order]
                sorted_pop = pop[order]
                
                # Update best
                if sorted_fits[0] < best_fx:
                    best_fx = sorted_fits[0]
                    best_x = sorted_pop[0].copy()
                
                # Update mean
                m_new = m + sigma * np.sum(w[:, np.newaxis] * sorted_pop[:mu], axis=0)
                
                # Update step size
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv @ (m_new - m) / sigma)
                
                # Calculate hsig
                # Ensure denominator is not zero
                denom = np.sqrt(1 - (1 - cs)**((self.calls / (cs * lambda_))**2)) if (self.calls / (cs * lambda_) > 1) else 1.0
                if denom < 1e-10:
                    denom = 1e-10
                hsig = (np.linalg.norm(ps) / denom < (1.4 + 2 / (n + 1)) * chiN) if (self.calls / (cs * lambda_) > 1) else True
                
                # Update sigma
                sigma = sigma * np.exp((cs / ds) * (np.linalg.norm(ps) / chiN - 1))
                # Clamp sigma to prevent explosion or collapse
                sigma = np.clip(sigma, 1e-10, 1e10)

                # Update covariance
                rank_mu = np.outer(m_new - m, m_new - m) / (sigma**2 if sigma > 0 else 1.0)
                rank_one = np.outer(ps, ps)
                delta_hsig = (1 - hsig) * cc * (1 + cc)
                
                C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
                
                # Ensure C is positive definite
                # Add small regularization if needed
                C += 1e-12 * np.eye(n)
                
                try:
                    C_inv = np.linalg.inv(C)
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If Cholesky fails, reset C to identity or add more regularization
                    C = np.eye(n)
                    C_inv = np.eye(n)
                    L = np.eye(n)
                    sigma *= 0.5
                
                m = m_new
                
        except Exception:
            # If any unexpected error occurs, return best found so far
            pass

        return best_x, best_fx
