import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        
        try:
            # CMA-ES parameters
            # Population size
            lam = 4 + int(3 * np.log(dim))  # Standard CMA-ES population size
            mu = lam // 2  # Number of best individuals to update weights
            
            # Weights for the best mu individuals
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            
            # Effective population size
            mueff = 1.0 / np.sum(weights**2)
            
            # Learning rates
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            
            # Damping for covariance matrix update
            dampsigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            
            # Initial step size
            sigma = 2.0
            
            # Mean vector (start at origin, which is in the center)
            mean = np.zeros(dim)
            
            # Covariance matrix (identity)
            C = np.eye(dim)
            
            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Precompute Cholesky decomposition
            C_sqrt = np.linalg.cholesky(C)
            
            # Track best
            # Initial evaluation at origin
            fx = func(mean)
            calls_made += 1
            best_f = fx
            best_x = mean.copy()
            
            # Main loop
            while calls_made < budget and time.time() < deadline:
                # Generate population
                offspring = []
                for i in range(lam):
                    if calls_made >= budget or time.time() >= deadline:
                        break
                    # Sample from N(mean, sigma^2 * C)
                    z = rng.standard_normal(dim)
                    x_off = mean + sigma * (C_sqrt @ z)
                    x_off = np.clip(x_off, -5, 5)
                    offspring.append(x_off)
                    calls_made += 1
                    fx_off = func(x_off)
                    offspring[-1] = (x_off, fx_off)
                
                if not offspring:
                    break
                
                # Sort by fitness
                offspring.sort(key=lambda x: x[1])
                
                # Update best
                if offspring[0][1] < best_f:
                    best_f = offspring[0][1]
                    best_x = offspring[0][0].copy()
                
                # Update mean
                # Use weighted average of the best mu
                x_w = np.zeros(dim)
                for i in range(min(mu, len(offspring))):
                    x_w += weights[i] * offspring[i][0]
                
                # Update evolution path for sigma
                C_sqrt_inv = np.linalg.inv(C_sqrt)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_sqrt_inv @ (x_w - mean) / sigma
                
                # Hessian norm
                hsig = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim + 1 - 1) else 0.0
                
                # Update evolution path for covariance
                p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * (x_w - mean) / sigma
                
                # Update mean
                mean = x_w
                
                # Update covariance matrix
                rank_one = np.outer(p_sigma, p_sigma)
                rank_mu = np.zeros((dim, dim))
                for i in range(min(mu, len(offspring))):
                    y_i = C_sqrt_inv @ (offspring[i][0] - mean + sigma * (x_w - mean))  # Approximation, standard CMA uses (x_i - mean)
                    # Actually, standard is: y_i = (x_i - mean_old) / sigma
                    # Let's use the simpler update:
                    y_i = (offspring[i][0] - mean) / sigma
                    rank_mu += weights[i] * np.outer(y_i, y_i)
                
                C = (1 + c_1 * (2 - c_1) - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                
                # Ensure positive semi-definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Update sigma
                sigma = sigma * np.exp(c_sigma / dampsigma * (np.linalg.norm(p_sigma) / np.sqrt(dim + 1 - 1) - 1))
                
                # Update Cholesky
                C_sqrt = np.linalg.cholesky(C)
                
                # Check for convergence
                if sigma < 1e-6 or np.max(eigvals) / np.min(eigvals) > 1e14:
                    break

        except Exception:
            pass

        return best_x, best_f
