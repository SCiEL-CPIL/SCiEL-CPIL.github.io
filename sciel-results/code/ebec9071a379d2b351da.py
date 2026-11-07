import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        if lam < 4:
            lam = 4
        # For dim=5, lam = 4 + 3*1.609 = 8.8 -> 8. Let's use 12 for better exploration.
        lam = 12
        
        # Initial mean
        # Random start within [-5, 5]
        m = np.random.uniform(-5, 5, dim)
        
        # Initial step size
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Pre-compute constants
        mu = lam // 2  # Number of selected points
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Cholesky factor of C
        C_sqrt = np.linalg.cholesky(C)
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = m
        best_f = func(m)
        calls_used = 1
        
        while calls_used < budget:
            # Check time limit (safety margin)
            if time.time() - start_time > 280:
                break
                
            # Generate lambda points
            ys = np.array([np.random.multivariate_normal(np.zeros(dim), C, 1)[0] for _ in range(lam)])
            # More efficient: 
            # ys = np.random.randn(lam, dim) @ C_sqrt.T
            # But multivariate_normal is fine for dim=5, lam=12.
            
            xs = np.array([m + sigma * y for y in ys])
            
            # Clip to bounds
            xs = np.clip(xs, -5, 5)
            
            # Evaluate
            fs = np.array([func(x) for x in xs])
            calls_used += lam
            
            # Sort by fitness
            idx = np.argsort(fs)
            xs_sorted = xs[idx]
            fs_sorted = fs[idx]
            
            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = xs_sorted[0]
                
            # Update mean
            m_new = np.sum(w[:, np.newaxis] * xs_sorted[:mu], axis=0)
            
            # Update sigma (CSA)
            # y_sigma = (m_new - m) / sigma
            y_sigma = (m_new - m) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Limit sigma
            if sigma > 100: sigma = 100
            if sigma < 1e-12: sigma = 1e-12
            
            # Update covariance
            # p_c update
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - m) / sigma
            
            # Rank 1 and Rank mu update
            v = m_new - m
            C = (1 + c_c * (2 - c_c) * c_sigma - c_1 - c_mu) * C \
                + c_1 * np.outer(v, v) / sigma**2 \
                + c_mu * np.sum(w[:, np.newaxis, np.newaxis] * np.outer(xs_sorted[:mu] - m, xs_sorted[:mu] - m), axis=0) / sigma**2
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update Cholesky factor
            # If C is not positive definite, add small identity
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-6 * np.eye(dim)
                C_sqrt = np.linalg.cholesky(C)
                
            m = m_new
            
            # Convergence check
            if sigma < 1e-10:
                break

        return best_x, best_f
