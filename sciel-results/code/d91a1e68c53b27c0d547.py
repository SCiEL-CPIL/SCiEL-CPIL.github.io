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
        deadline = start_time + 290  # Leave 10s margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        # For n=5, lambda=9 is slightly more efficient than 10
        lam = 9
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
        # Start near the center
        x_mean = np.zeros(dim)
        sigma = 2.5  # Initial step size, slightly smaller for better early convergence
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = np.inf
        
        evals_used = 0
        
        # Helper to decompose C and get sqrt_inv
        def get_cholesky(C):
            # Add regularization for numerical stability
            C_reg = C + 1e-10 * np.eye(dim)
            try:
                L = np.linalg.cholesky(C_reg)
                return L
            except np.linalg.LinAlgError:
                # Fallback: use eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(C_reg)
                eigvals = np.clip(eigvals, 1e-12, None)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
                return L

        L = get_cholesky(C)
        
        # Initial evaluation of the mean
        f_mean = func(x_mean)
        evals_used += 1
        if f_mean < best_f:
            best_f = f_mean
            best_x = x_mean.copy()
        
        # Convergence detection
        no_improvement_count = 0
        max_no_improvement = 15  # Stop CMA-ES if no improvement for 15 iterations
        
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
            
            # Check for improvement
            if fs[0] < best_f - 1e-10:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Stop if no improvement for too many iterations
            if no_improvement_count >= max_no_improvement:
                break
            
            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(w[:mu, None] * xs[:mu], axis=0)
            
            # Compute C^{-1/2} using eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Update evolution path for sigma control
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            
            # Hessian potential norm
            norm_ps = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2))
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            
            # Update evolution path for covariance control
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            
            # Update covariance matrix
            sum_w_y = np.zeros((dim, dim))
            for i in range(mu):
                sum_w_y += w[i] * np.outer(ys[i], ys[i])
                
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * C) + c_mu * sum_w_y
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(dim)
            
            # Update Cholesky factor
            L = get_cholesky(C)
            
        # Local refinement using (1+1)-ES with remaining budget
        remaining = budget - evals_used
        if remaining > 0 and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f
            # Use the current CMA-ES sigma as a starting point for local step size
            sigma_local = max(sigma, 0.1)
            
            # Adaptive (1+1)-ES with restarts
            stagnant = 0
            max_stagnant = 30
            
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
                    stagnant = 0
                    # Increase step size on success
                    sigma_local *= 1.05
                else:
                    stagnant += 1
                    # Decrease step size on failure
                    sigma_local *= 0.95
                
                # If stagnant for too long, restart with larger step
                if stagnant >= max_stagnant:
                    sigma_local = max(sigma_local * 2.0, 0.5)
                    stagnant = 0
                
                # Ensure sigma_local stays in reasonable bounds
                sigma_local = np.clip(sigma_local, 1e-5, 5.0)
                    
        return best_x, best_f
