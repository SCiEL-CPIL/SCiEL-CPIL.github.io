import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time management
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill

        # CMA-ES Parameters
        # Population size: lambda = 4 + floor(3 * log(10 + dim))
        # For dim=5: 4 + 3*log(15) ≈ 4 + 3*1.17 = 7.5 -> 8
        lam = 4 + int(3 * np.log(10 + dim))
        
        # Initial parameters
        x0 = np.zeros(dim)
        sigma0 = 2.0  # Initial step size, covering a good portion of [-5,5]
        
        # CMA-ES state
        x_mean = x0.copy()
        sigma = sigma0
        C = np.eye(dim)  # Covariance matrix
        C_inv = np.eye(dim)
        
        # Adaptation parameters
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 2 / dim) / (dim + 2 + mu_eff - 2)))
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Damping for covariance update
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        # Helper to evaluate and update best
        def evaluate(x):
            nonlocal best_x, best_f, calls
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f, x_clipped

        # Initial evaluation
        f0, x0_clipped = evaluate(x0)
        x_mean = x0_clipped.copy()
        
        # Main loop
        while calls < budget:
            # Time check
            if time.time() - start_time > time_limit:
                break
            
            # Generate population
            population = []
            fitnesses = []
            
            for i in range(lam):
                # Sample from N(x_mean, sigma^2 * C)
                # Use Cholesky or eigen decomposition for C
                # For small dim, eigendecomposition is robust
                try:
                    w, V = np.linalg.eigh(C)
                    w = np.clip(w, 1e-12, None)
                    sqrt_w = np.sqrt(w)
                    z = rng.standard_normal(dim)
                    y = V @ (sqrt_w * z)
                    x_i = x_mean + sigma * y
                except:
                    x_i = x_mean + sigma * rng.standard_normal(dim)
                
                f_i, x_i_clipped = evaluate(x_i)
                if calls >= budget:
                    break
                    
                population.append(x_i_clipped)
                fitnesses.append(f_i)
            
            if calls >= budget:
                break
            
            # Sort by fitness
            indices = np.argsort(fitnesses)
            population = [population[i] for i in indices]
            fitnesses = [fitnesses[i] for i in indices]
            
            # Update mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * population[i]
            
            # Update step size (sigma)
            x_old_mean = x_mean.copy()
            x_mean = x_new
            
            # Evolution path for sigma
            try:
                C_sqrt_inv = V @ np.diag(1.0 / sqrt_w) @ V.T
            except:
                C_sqrt_inv = np.eye(dim)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ ((x_mean - x_old_mean) / sigma)
            
            # Hessian potential condition for sigma update
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - lam))) < (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance matrix
            # p_c path
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_sqrt_inv @ ((x_mean - x_old_mean) / sigma)
            
            # Rank-mu and rank-1 updates
            delta_dist = np.heaviside(np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - lam))) - (1.4 + 2 / (dim + 1)) * chi_n, 1.0)
            
            C = (1 + c_1 * delta_dist * (mu_eff - 1) + c_mu) * C
            C -= c_1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            for i in range(mu):
                y_i = (population[i] - x_old_mean) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small regularization to ensure invertibility
            C += 1e-12 * np.eye(dim)
            
            try:
                C_inv = np.linalg.inv(C)
            except:
                C_inv = np.eye(dim)
                C = np.eye(dim)
            
            # Optional: Check for stagnation and restart
            # If sigma is very small and no progress, we might need to restart
            # But CMA-ES usually handles this. If stuck, we can inject diversity.
            if sigma < 1e-4 and (calls % 100 == 0):
                # Small restart: jitter mean
                x_mean = x_mean + rng.normal(0, 0.1, dim)
                sigma = 0.5
                C = np.eye(dim)
                C_inv = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
        
        return best_x, best_f
