import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard
        start_time = time.time()
        deadline = start_time + 280 # Leave 20s margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        # For n=5, lambda=10 is standard and efficient
        lam = 10
        mu = lam // 2
        
        # Weights for the top mu individuals
        # Standard weights: w_i = log(mu + 0.5) - log(i)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        w = np.concatenate([w, np.zeros(lam - mu)])
        
        # Variance-effective population size
        mueff = 1.0 / np.sum(w**2)
        
        # Strategy parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 - mueff / (dim + 2) + c_c) / ((dim + 1.3) ** 2 + mueff + c_c)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Initialization
        # Start near the center, but slightly randomized to avoid symmetry issues
        x_mean = np.zeros(dim)
        sigma = 3.0 # Initial step size, covering a good portion of [-5,5]
        
        # Covariance matrix
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute sqrt(C) inverse? We use Cholesky decomposition each step for simplicity and robustness
        # For small dim, this is fast.
        
        best_x = x_mean.copy()
        best_f = np.inf
        
        evals_used = 0
        
        # Helper to decompose C and get sqrt_inv
        def get_sqrt_inv(C):
            # C is positive definite
            L = np.linalg.cholesky(C)
            # We need C^{-1/2}
            # C = L L^T
            # C^{-1} = L^{-T} L^{-1}
            # C^{-1/2} = L^{-1} (L^{-T} L^{-1})^{1/2} ... simpler to use eigendecomposition or just inverse of cholesky
            # Actually, for sampling y ~ N(0, C), we can sample z ~ N(0, I) and y = L z.
            # For updating C, we need C^{-1}.
            C_inv = np.linalg.inv(C)
            # For the step size control, we need ||p_sigma|| / sqrt(1 - (1-c_c)^2)
            # And we need to sample from N(0, C).
            return L, C_inv

        L, C_inv = get_sqrt_inv(C)
        
        # Initial evaluation of the mean
        # Note: The problem says func clips. We can evaluate x_mean.
        f_mean = func(x_mean)
        evals_used += 1
        if f_mean < best_f:
            best_f = f_mean
            best_x = x_mean.copy()
            
        while evals_used < budget:
            if time.time() > deadline:
                break
                
            # Check if we have enough budget for a full population
            if budget - evals_used < lam:
                break

            # Generate offspring
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                if evals_used >= budget or time.time() > deadline:
                    break
                    
                # Sample from N(0, I)
                z = rng.standard_normal(dim)
                # Transform to N(0, C)
                y = L @ z
                # New position
                x_new = x_mean + sigma * y
                # Clip to domain
                x_new = np.clip(x_new, -5, 5)
                
                f_val = func(x_new)
                evals_used += 1
                
                ys.append(y)
                xs.append(x_new)
                fs.append(f_val)
                
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_new.copy()
                    
            if len(xs) < lam:
                break
                
            ys = np.array(ys)
            xs = np.array(xs)
            fs = np.array(fs)
            
            # Sort by fitness
            idx = np.argsort(fs)
            xs = xs[idx]
            ys = ys[idx]
            fs = fs[idx]
            
            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(w[:mu, None] * xs[:mu], axis=0)
            
            # Update evolution path for sigma control
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma(2-c_sigma)mueff) * C^{-1/2} (x_mean - x_mean_old) / sigma
            # We need C^{-1/2}. We have L such that C = L L^T.
            # C^{-1/2} is not exactly L^{-1}.
            # However, for the path update, we often use the "natural" gradient direction.
            # A simpler approximation often used in implementations:
            # Let's compute C_inv_sqrt properly or use the approximation.
            # For small dim, let's compute eigendecomposition of C to get C^{-1/2}
            eigvals, eigvecs = np.linalg.eigh(C)
            # Avoid division by zero
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            
            # Hessian potential norm
            norm_ps = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2))
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            
            # Update evolution path for covariance control
            # p_c = (1 - c_c) p_c + sqrt(c_c(2-c_c)) * C^{-1/2} (x_mean - x_mean_old) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            
            # Update covariance matrix
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - (1-c_c)^2) C) + c_mu sum w_i (y_i y_i^T)
            # Note: The term (x_mean - x_mean_old)/sigma is related to y_bar
            # Let y_bar = (x_mean - x_mean_old) / sigma
            y_bar = (x_mean - x_mean_old) / sigma
            
            # Weighted sum of outer products of selected y's
            # y_selected = ys[:mu]
            # sum_w_y = sum(w_i * y_i * y_i^T)
            sum_w_y = np.zeros((dim, dim))
            for i in range(mu):
                sum_w_y += w[i] * np.outer(ys[i], ys[i])
                
            # Correction term for the rank-one update of C
            # The standard formula involves C itself.
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * C) + c_mu * sum_w_y
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            # Add small regularization if needed
            C += 1e-12 * np.eye(dim)
            
            # Update L and C_inv_sqrt for next iteration
            L, C_inv = get_sqrt_inv(C)
            
            # Optional: If sigma becomes too small or too large, adjust?
            # CMA-ES handles this via the path.
            
            # If the population has collapsed (all points identical), restart?
            # With 500 evals, probably not necessary.
            
        # Local refinement using (1+1)-ES with remaining budget
        remaining = budget - evals_used
        if remaining > 0 and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f
            # Use the current CMA-ES sigma as a starting point for local step size
            # If sigma is very small, it means we are converged. If large, we are not.
            # A safe bet is to use a fraction of the domain size or the current sigma.
            sigma_local = max(sigma, 1e-3)
            
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                
                z = rng.standard_normal(dim)
                x_new = x_local + sigma_local * z
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                evals_used += 1
                
                if f_new < f_local:
                    f_local = f_new
                    x_local = x_new.copy()
                    best_f = f_local
                    best_x = x_local.copy()
                    # Slightly increase step size on success? Or keep it?
                    # Standard (1+1) often keeps sigma or increases slightly.
                    # For f7, which has flat regions, increasing might help escape local plateaus.
                    sigma_local *= 1.1
                else:
                    # Decrease step size on failure
                    sigma_local *= 0.9
                    
                # Ensure sigma_local doesn't get too small
                if sigma_local < 1e-6:
                    sigma_local = 1e-6
                    
        return best_x, best_f
