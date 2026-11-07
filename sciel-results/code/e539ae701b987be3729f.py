import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0  # Safety margin for 300s limit
        d = self.dim
        budget = self.budget

        # 1. Global Phase: Sobol Sampling
        # We use a low-discrepancy sequence to sample the space efficiently.
        # This helps in finding the "step" region of f7 more reliably than random.
        n_global = max(10, budget // 5)  # Use 20% of budget for global search
        if n_global > budget - 10:
            n_global = max(1, budget - 10)

        # Generate Sobol points in [0, 1]^d
        # We implement a simple Sobol sampler or use a fallback if scipy is not available.
        # Since scipy is available, we use it for robustness.
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d, scramble=True, seed=42)
            points = sampler.random(n_global)
        except ImportError:
            # Fallback to Halton or Random if scipy is missing (unlikely)
            points = np.random.rand(n_global, d)

        # Map [0, 1]^d to [-5, 5]^d
        points = points * 10.0 - 5.0

        best_x = None
        best_f = np.inf
        calls_made = 0

        # Evaluate global points
        for i in range(n_global):
            if time.time() - start_time > timeout:
                break
            x = points[i]
            try:
                fx = func(x)
            except Exception:
                fx = np.inf
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            calls_made += 1

        # 2. Local Phase: CMA-ES
        # CMA-ES is highly effective for non-separable functions like f7.
        # We start from the best point found in the global phase.
        
        # Implementing a basic CMA-ES loop manually to avoid external dependencies 
        # other than numpy, or using scipy if available. 
        # For 5D, a manual implementation is feasible and fast.
        
        # CMA-ES Parameters
        pop_size = max(4, int(4 + 3 * np.log(d)))  # Standard population size
        sigma = 0.5  # Initial step size
        x_mean = best_x.copy()
        
        # Covariance matrix C
        C = np.eye(d)
        # Step size for covariance update
        c_cov = (4 + np.log(d)) / (d + 4)
        mu = pop_size // 2  # Number of recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size adaptation
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (d + 1)) + c_sigma
        
        # Memory for path updates
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Boundary handling: We will clip and if a point is out of bounds, 
        # we reflect it or just clip. Clipping is simpler but can bias.
        # For f7, the minimum is inside [-4, 4]^5, so clipping is safe.
        
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f

        # We will run CMA-ES until budget or timeout
        # Each generation consumes pop_size calls
        while calls_made + pop_size <= budget:
            if time.time() - start_time > timeout:
                break

            # Generate offspring
            Y = np.zeros((pop_size, d))
            for i in range(pop_size):
                # Sample from N(0, C)
                z = np.random.multivariate_normal(np.zeros(d), C)
                y = x_mean + sigma * z
                # Clip to box
                y = np.clip(y, -5.0, 5.0)
                Y[i] = y

            # Evaluate offspring
            F = np.zeros(pop_size)
            for i in range(pop_size):
                try:
                    F[i] = func(Y[i])
                except Exception:
                    F[i] = np.inf
                calls_made += 1
                if time.time() - start_time > timeout:
                    break
            
            if time.time() - start_time > timeout:
                break

            # Sort by fitness
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = Y[0].copy()

            # Update mean
            x_mean_new = np.sum(weights[:, np.newaxis] * Y[:mu], axis=0)
            
            # Update step size sigma
            # Compute y_sigma
            C_inv_sqrt = np.linalg.inv(np.sqrtm(C))
            y_sigma = (C_inv_sqrt @ (x_mean_new - x_mean).T).T / sigma
            norm_y_sigma = np.linalg.norm(y_sigma)
            h_sigma = 1.0 if (norm_y_sigma / np.sqrt(1 - (1 + 2 * c_sigma / d_sigma) ** (-calls_made)) < (1 + 2 * c_sigma / d) * np.sqrt((d + 1) / 2 - 1)) else 0.0
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * h_sigma * y_sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update covariance matrix
            p_c = (1 - c_cov) * p_c + c_cov * h_sigma * (x_mean_new - x_mean) / sigma
            delta_h = (h_sigma - 0.5) * (1 + c_cov)
            C = (1 + c_cov * (mu_eff - 1) + 2 * c_cov * delta_h) * C + c_cov * (p_c @ p_c.T - C)
            # Symmetrize and ensure positive definite
            C = (C + C.T) / 2
            # Ensure eigenvalues are positive
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            x_mean = x_mean_new
            
            # If sigma becomes too small or too large, we might stop or reset
            if sigma < 1e-6:
                break

        return best_x, best_f
