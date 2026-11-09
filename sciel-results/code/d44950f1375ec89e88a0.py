import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lmbda = max(4 + int(3 * np.log(n)), 4)
        mu = lmbda // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = 1.0 / (lmbda + 3.0)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 2.0) / (lmbda + 2.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / lmbda) / (lmbda + 4.0 + 2.0 * mu_eff / lmbda)
        c_1 = 2.0 / ((lmbda + 1.3) ** 2 + n)
        c_mu = min(1.0 - c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((lmbda + 2.0) ** 2 + n))
        
        rng = np.random.default_rng(0)
        x_mean = rng.uniform(-5, 5, n)
        sigma = 1.0
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = x_mean.copy()
        best_f = func(x_mean)
        budget_consumed = 1
        
        while budget_consumed + lmbda <= budget:
            # Compute Cholesky once per generation
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize if not PD
                C = C + 1e-6 * np.eye(n)
                L = np.linalg.cholesky(C)
                
            # Sample
            zs = rng.standard_normal((lmbda, n))
            xs = x_mean + sigma * (zs @ L.T)
            xs = np.clip(xs, -5, 5)
            
            # Evaluate
            fs = np.array([func(x) for x in xs])
            budget_consumed += lmbda
            
            # Sort
            idx = np.argsort(fs)
            xs_sorted = xs[idx]
            fs_sorted = fs[idx]
            
            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = xs_sorted[0].copy()
                
            # Update mean
            x_new_mean = np.sum(weights[:, np.newaxis] * xs_sorted[:mu], axis=0)
            
            # Update step size
            y_sigma = (x_new_mean - x_mean) / sigma
            p_sigma = (1.0 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * y_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1.0))
            
            # Update covariance
            y_c = (x_new_mean - x_mean) / sigma
            p_c = (1.0 - c_c) * p_c + np.sqrt(c_c * (2.0 - c_c) * mu_eff) * y_c
            
            # Rank-1 update
            C = (1.0 - c_1 - c_mu) * C + c_1 * (p_c @ p_c + (1.0 - c_c) * (2.0 - c_c) * np.eye(n))
            
            # Rank-mu update
            # Sum of weighted outer products of (x_i - x_mean)
            # Note: x_mean here is the OLD mean for the update calculation in standard CMA-ES
            diff = xs_sorted[:mu] - x_mean
            rank_mu = np.sum(weights[:, np.newaxis] * (diff * diff.T), axis=0)
            C += c_mu * rank_mu
            
            x_mean = x_new_mean
            
            # Termination
            if sigma < 1e-8:
                break
                
        return best_x, best_f
