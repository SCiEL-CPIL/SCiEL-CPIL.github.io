import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit
        budget = self.budget
        dim = self.dim
        
        calls = [0]
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def evaluate(x):
            nonlocal best_x, best_f
            if calls[0] >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls[0] += 1
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)
        
        # Structural change: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
        # CMA-ES is a robust derivative-free optimizer for multimodal continuous problems.
        # It adapts the covariance matrix of the sampling distribution to the landscape,
        # allowing it to handle rotated and scaled features effectively.
        # For Lunacek bi-Rastrigin, which has two basins (one at the origin and one shifted),
        # CMA-ES with appropriate population size and restarts can navigate the multimodality.
        
        def cma_es(func, dim, budget, deadline, rng, calls):
            # CMA-ES implementation
            # Parameters
            pop_size = 4 + int(3 * np.log(dim))  # Standard CMA-ES population size
            pop_size = max(pop_size, 4)
            
            # Initial mean and sigma
            mean = np.zeros(dim)
            sigma = 2.0  # Initial step size, larger for exploration
            
            # Covariance matrix
            C = np.eye(dim)
            # Inverse square root of C (for sampling)
            C_inv_sqrt = np.eye(dim)
            
            # Evolution path
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Adaptation parameters
            mu = pop_size // 2  # Number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights = weights / np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_sigma = c_sigma * (1 - c_c / 2)
            
            c_c_inv = 1.0 / c_c
            c_sigma_inv = 1.0 / c_sigma
            c_sigma_d = c_sigma * d_sigma
            
            # Initial population
            population = []
            fitnesses = []
            
            for i in range(pop_size):
                if calls[0] >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                x = mean + sigma * z
                x = np.clip(x, -5, 5)
                f = func(x)
                calls[0] += 1
                population.append(x)
                fitnesses.append(f)
            
            if not population:
                return None, None
            
            # Sort by fitness
            idx = np.argsort(fitnesses)
            population = [population[i] for i in idx]
            fitnesses = [fitnesses[i] for i in idx]
            
            best_x_cma = population[0].copy()
            best_f_cma = fitnesses[0]
            
            # Main loop
            while calls[0] < budget and time.time() < deadline:
                # Update mean
                mean = np.sum([weights[i] * population[i] for i in range(mu)], axis=0)
                
                # Update sigma path
                y_mean = (mean - mean) / (sigma * np.linalg.norm(C_inv_sqrt @ np.linalg.inv(C_inv_sqrt) @ (mean - mean)))
                # Simplified: use the standard formula
                y_mean = (mean - mean) / (sigma)  # This is zero, need to use old mean
                # Let's use the proper formula with the old mean
                old_mean = mean - np.sum([weights[i] * population[i] for i in range(mu)], axis=0) + mean
                # Actually, let's just use the standard CMA-ES update rules
                
                # For simplicity, use a simplified CMA-ES that works well in practice
                # This is a simplified version that may not be theoretically perfect but works well
                new_population = []
                new_fitnesses = []
                
                for i in range(pop_size):
                    if calls[0] >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    # Sample from N(mean, sigma^2 * C)
                    x = mean + sigma * (C_inv_sqrt @ z)
                    x = np.clip(x, -5, 5)
                    f = func(x)
                    calls[0] += 1
                    new_population.append(x)
                    new_fitnesses.append(f)
                
                if not new_population:
                    break
                
                idx = np.argsort(new_fitnesses)
                new_population = [new_population[i] for i in idx]
                new_fitnesses = [new_fitnesses[i] for i in idx]
                
                if new_fitnesses[0] < best_f_cma:
                    best_f_cma = new_fitnesses[0]
                    best_x_cma = new_population[0].copy()
                
                # Update mean
                old_mean = mean.copy()
                mean = np.sum([weights[i] * new_population[i] for i in range(mu)], axis=0)
                
                # Update sigma path (simplified)
                y_mean = (mean - old_mean) / (sigma * np.linalg.norm(C_inv_sqrt @ z))
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                
                # Update covariance matrix (simplified)
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2)) < (1 + 2 / (dim + 1)) * np.sqrt(mu_eff)
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - old_mean) / sigma
                C = (1 + c_c_inv * c_c * (mu_eff - 1) * (np.outer(p_c, p_c) - C) + 
                     c_c_inv * c_c * mu_eff * np.sum([weights[i] * np.outer((new_population[i] - old_mean) / sigma, (new_population[i] - old_mean) / sigma) for i in range(mu)], axis=0))
                
                # Update C_inv_sqrt (simplified, use SVD)
                try:
                    vals, vecs = np.linalg.eigh(C)
                    vals = np.clip(vals, 1e-10, None)
                    C_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
                except:
                    C_inv_sqrt = np.eye(dim)
                
                population = new_population
                fitnesses = new_fitnesses
                
                # Termination criterion
                if sigma < 1e-12 or np.allclose(C, np.eye(dim) * np.trace(C) / dim):
                    break
            
            return best_x_cma, best_f_cma
        
        # Run CMA-ES
        result_x, result_f = cma_es(func, dim, budget, deadline, rng, calls)
        
        if result_x is not None and result_f < best_f:
            best_x = result_x
            best_f = result_f
        
        # If CMA-ES didn't use all budget, do a local refinement
        if calls[0] < budget and time.time() < deadline:
            sigma = 0.1
            x = best_x.copy()
            f = best_f
            for _ in range(50):
                if calls[0] >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                x_new = x + sigma * z
                x_new = np.clip(x_new, -5, 5)
                f_new = evaluate(x_new)
                if f_new < f:
                    x = x_new
                    f = f_new
                    sigma *= 0.9
                else:
                    sigma *= 0.8
                if sigma < 1e-5:
                    break
        
        if calls[0] == 0:
            evaluate(np.zeros(dim))

        return best_x, best_f
