import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = float('inf')
        self.calls = 0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        dim = self.dim
        budget = self.budget
        
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)

        # Helper to evaluate and track
        def evaluate(x):
            if self.calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5, 5)
            self.calls += 1
            f = func(x)
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        # --- Phase 1: Gaussian Process Exploration ---
        
        # GP Hyperparameters
        sigma_f = 1.0  # Signal std
        length_scale = 1.0  # RBF length scale
        sigma_n = 1e-6  # Noise std
        
        def rbf_kernel(X1, X2, l):
            # Compute squared distances
            # X1: (n, d), X2: (m, d)
            sq_dists = np.sum(X1**2, 1)[:, None] + np.sum(X2**2, 1)[None, :] - 2 * X1.dot(X2.T)
            sq_dists = np.maximum(sq_dists, 0) # Numerical stability
            return np.exp(-0.5 * sq_dists / (l**2))
        
        # Store observed data
        X_obs = []
        y_obs = []
        
        # Initial random sampling
        n_init = 10
        for _ in range(n_init):
            x = np.random.uniform(-5, 5, dim)
            f = evaluate(x)
            X_obs.append(x)
            y_obs.append(f)
            
        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)
        
        # GP Prediction
        def gp_predict(X_new):
            # X_new: (m, d)
            K = rbf_kernel(X_obs, X_new, length_scale)
            K_obs = rbf_kernel(X_obs, X_obs, length_scale) + sigma_n * np.eye(len(X_obs))
            
            # Solve K_obs @ alpha = y_obs
            try:
                alpha = np.linalg.solve(K_obs, y_obs)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if singular
                alpha = np.linalg.pinv(K_obs) @ y_obs
                
            mu = K.T @ alpha
            K_obs_inv = np.linalg.inv(K_obs)
            K_new_new = rbf_kernel(X_new, X_new, length_scale)
            var = sigma_f**2 - np.sum(K @ K_obs_inv @ K.T, axis=1)
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)
            return mu, std
        
        # Expected Improvement
        def expected_improvement(mu, std, best_f):
            # EI = (best_f - mu) * Phi((best_f - mu) / std) + std * phi((best_f - mu) / std)
            z = (best_f - mu) / std
            # Clip z to avoid overflow in exp
            z = np.clip(z, -10, 10)
            from scipy.stats import norm
            return (best_f - mu) * norm.cdf(z) + std * norm.pdf(z)
        
        # GP Loop
        n_gp_steps = 50
        for _ in range(n_gp_steps):
            if self.calls >= budget or time.time() > deadline:
                break
                
            # Predict on a set of candidate points
            # Sample 1000 random points
            n_cand = 1000
            X_cand = np.random.uniform(-5, 5, (n_cand, dim))
            
            mu, std = gp_predict(X_cand)
            
            # Calculate EI
            ei = expected_improvement(mu, std, self.best_f)
            
            # Select point with highest EI
            best_idx = np.argmax(ei)
            x_next = X_cand[best_idx]
            
            f_next = evaluate(x_next)
            
            # Update observations
            X_obs = np.vstack([X_obs, x_next])
            y_obs = np.append(y_obs, f_next)
            
            # Optional: If EI is very small, we might be converged, but let's stick to budget for now.
            # Or break if max EI < threshold
            if np.max(ei) < 1e-4:
                break

        # --- Phase 2: CMA-ES Local Optimization ---
        
        # Initialize CMA-ES at the best point found
        x_mean = self.best_x.copy()
        
        # Choose initial sigma
        # If we are close to the minimum, a smaller sigma is better.
        # Let's estimate the local scale from the last few GP points or just use a fixed small value.
        sigma = 0.5
        
        # CMA-ES Parameters
        lam = 9
        mu = 4
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))
        
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Main CMA-ES Loop
        while self.calls < budget:
            if time.time() > deadline:
                break
                
            n_off = min(lam, budget - self.calls)
            if n_off <= 0:
                break
            
            # Sample offspring
            # Use Cholesky for speed
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                L = np.eye(dim)
                
            offspring = []
            for i in range(n_off):
                z = np.random.randn(dim)
                y = x_mean + sigma * (L @ z)
                fy = evaluate(y)
                offspring.append((y, fy))
                
            if time.time() > deadline:
                break
                
            # Sort
            offspring.sort(key=lambda t: t[1])
            
            # Update Mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += w[i] * offspring[i][0]
            x_new_mean = y_w
            
            # Check for stagnation
            diff = x_new_mean - x_mean
            norm_diff = np.linalg.norm(diff)
            
            if norm_diff < 1e-12:
                sigma *= 1.5
                if sigma > 10:
                    break
                continue
                
            # Update Paths
            # C_inv_sqrt
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
            except:
                C_inv_sqrt = np.eye(dim)
                
            p_sigma = (1 - c_sigma) * p_sigma + \
                      np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ diff) / sigma
            
            h_sigma = 1.0
            if np.linalg.norm(p_sigma) > chi_n * 2.0:
                 h_sigma = 0.0
                 p_sigma = np.zeros(dim)
                 p_c = np.zeros(dim)
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ diff) / sigma
            
            # Update Sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update Covariance
            c1_term = c_1 * np.outer(p_c, p_c)
            
            cmu_term = np.zeros((dim, dim))
            for i in range(mu):
                y_i = offspring[i][0]
                diff_i = y_i - x_new_mean
                cmu_term += w[i] * np.outer(diff_i, diff_i)
            
            C = (1 - c_1 - c_mu + mu_eff * c_mu) * C + c1_term + c_mu * cmu_term
            C = (C + C.T) / 2
            
            x_mean = x_new_mean

        return self.best_x, self.best_f
