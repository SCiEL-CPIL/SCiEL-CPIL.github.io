"""
Diagonal-CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
Strategy:
  - Use Diagonal-CMA-ES to avoid expensive matrix operations and numerical instability.
  - Initialize the diagonal step sizes using PCA on a small set of random samples 
    to rapidly align the search scale with the principal axes of the landscape.
  - This is faster than full CMA-ES and robust for dim=5.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit

        dim = self.dim
        n_evals = 0
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')

        # CMA-ES parameters
        lambda_ = 8
        mu = lambda_ // 2
        
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)

        # Step size adaptation parameters (Diagonal)
        c_sigma = (mueff + 2) / ((dim + 2) ** 2 + mueff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Covariance adaptation parameters (Diagonal approx)
        # In Diagonal CMA-ES, we adapt the log of the diagonal entries.
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Initial state
        x_mean = np.zeros(dim)
        
        # Initial diagonal step sizes
        # We will refine these using PCA in the first generation
        sigma_diag = np.full(dim, 2.0)
        
        # Path for sigma adaptation (scalar in diagonal, but we keep a vector for consistency 
        # if we were to extend, but here we use a scalar path or per-dim paths? 
        # Standard Diagonal CMA-ES uses a single sigma or adapts the diagonal directly.
        # Let's use a simplified approach: adapt sigma_diag directly via rank-mu update on the log-diagonal.
        
        # Actually, a cleaner Diagonal CMA-ES formulation:
        # Let D be the diagonal matrix of step sizes.
        # y = x_mean + D * z
        # Update D using rank-mu updates on the covariance of y.
        
        # For simplicity and robustness, we will use a single global sigma 
        # and a diagonal scaling vector 's' such that step = sigma * s * z.
        # But standard Diagonal CMA-ES just adapts the diagonal of C.
        # Let's stick to adapting the diagonal of C directly.
        
        # Initialize C_diag
        C_diag = np.full(dim, 4.0) # sigma^2 = 4.0 -> sigma = 2.0
        
        # PCA Initialization
        # Sample 20 points to estimate the landscape shape
        init_samples = []
        init_vals = []
        for _ in range(20):
            x = rng.uniform(-5, 5, size=dim)
            f = func(x)
            n_evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            init_samples.append(x)
            init_vals.append(f)
        
        if n_evals >= self.budget:
            return best_x, best_f

        # Estimate initial diagonal variances from the spread of good points
        # Sort by fitness
        sorted_indices = np.argsort(init_vals)
        good_points = np.array([init_samples[i] for i in sorted_indices[:5]])
        # Center the good points
        good_center = np.mean(good_points, axis=0)
        good_centered = good_points - good_center
        # Covariance of good points
        cov_good = np.cov(good_centered, rowvar=False)
        # Use the diagonal of this covariance as a hint for initial anisotropy
        # Scale it to be reasonable (e.g., ensure min variance is not too small)
        init_variances = np.diag(cov_good)
        # Combine with prior (2.0^2 = 4.0)
        # If variance is small, trust the prior; if large, trust the data?
        # For Schaffers, the funnel is narrow in some directions.
        # Let's just use the geometric mean or clip the estimated variances.
        # A safer bet: use the estimated variances but clip them to [0.1, 10.0]
        C_diag = np.clip(init_variances, 0.1, 10.0)
        
        # Update mean to the center of good points
        x_mean = np.clip(good_center, -5, 5)

        p_sigma = np.zeros(dim)
        
        while n_evals < self.budget:
            if time.time() > deadline:
                break

            offspring = []
            offspring_vals = []
            
            for i in range(lambda_):
                if n_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                z = rng.standard_normal(dim)
                # Step is sqrt(C_diag) * z
                step = np.sqrt(C_diag) * z
                y = x_mean + step
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                n_evals += 1
                
                offspring.append(y)
                offspring_vals.append(fy)
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(offspring) < mu:
                break

            # Sort offspring
            indices = np.argsort(offspring_vals)
            sorted_offspring = np.array([offspring[i] for i in indices])
            
            # Update mean
            old_mean = x_mean.copy()
            x_mean = np.sum(weights[:, np.newaxis] * sorted_offspring[:mu], axis=0)
            x_mean = np.clip(x_mean, -5, 5)

            # Update C_diag (Diagonal Covariance)
            # Rank-mu update on the diagonal
            # y_i = (x_i - old_mean) / sqrt(C_diag)  (normalized steps)
            # C_new = (1 - c_1 - c_mu) * C_old + c_1 * p_c^2 + c_mu * sum(w_i * y_i^2)
            # In diagonal case, we just update the diagonal elements.
            
            # Path for covariance (simplified for diagonal)
            # We can use a path vector p_c
            p_c = np.zeros(dim) # Reset or maintain? Let's maintain for better adaptation
            
            # Actually, let's use a simpler update for Diagonal CMA-ES:
            # C_diag_new = (1 - c_1 - c_mu) * C_diag_old + c_mu * sum(w_i * (y_i)^2)
            # where y_i = (x_i - old_mean) / sqrt(C_diag_old)
            
            y_vecs = np.zeros((mu, dim))
            for i in range(mu):
                y_vecs[i] = (sorted_offspring[i] - old_mean) / np.sqrt(C_diag)
            
            # Weighted sum of squared normalized steps
            sum_yy = np.sum(weights[:, np.newaxis] * y_vecs ** 2, axis=0)
            
            # Update C_diag
            C_diag = (1 - c_1 - c_mu) * C_diag + c_mu * sum_yy
            
            # Add a small rank-one update to prevent collapse? 
            # The standard CMA-ES has a p_c term. For diagonal, we can approximate p_c
            # p_c = (1 - c_c) * p_c + h_c * sqrt(c_c * (2 - c_c) * mueff) * (C_sqrt_inv @ step)
            # C_sqrt_inv @ step = step / sqrt(C_diag)
            step = x_mean - old_mean
            h_c = 1.0 # Simplified
            
            # We need c_c. Let's use a standard value.
            c_c = (4 + mueff / dim) / (dim + 4) ** 2
            
            # p_c update
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mueff) * (step / np.sqrt(C_diag))
            
            # Rank-one update on diagonal
            C_diag = C_diag + c_1 * (p_c ** 2)
            
            # Ensure positive
            C_diag = np.maximum(C_diag, 1e-6)

            # Update sigma (global scaling if we used one, but here C_diag is the full step size)
            # In this formulation, C_diag IS the variance. So we don't have a separate sigma.
            # However, the step size adaptation (CSA) is important.
            # Let's add a global sigma that scales the whole thing?
            # No, if C_diag adapts, it handles the scaling.
            # But if all dimensions shrink, we might lose exploration.
            # Let's keep it simple: C_diag adapts the scale per dimension.

        return best_x, best_f
