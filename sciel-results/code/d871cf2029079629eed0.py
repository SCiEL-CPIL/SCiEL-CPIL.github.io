import numpy as np
import time
import sys

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = 325.0  # Safety margin from 330s limit
        
        # 1. Initial random sampling to find a good starting point and scale
        # Use 10% of budget for exploration
        n_init = max(10, self.budget // 10)
        best_x = None
        best_f = float('inf')
        xs = []
        fs = []
        
        # Random start
        rng = np.random.default_rng(42)
        for _ in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            f = func(x)
            xs.append(x)
            fs.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # 2. CMA-ES Search
        # Parameters
        n = self.dim
        # Population size
        lambda_pop = 4 + int(3 * np.log(n))
        # Remaining budget
        remaining_budget = self.budget - len(xs)
        max_generations = remaining_budget // lambda_pop
        
        if max_generations > 0:
            # Initialize CMA-ES state
            # Mean
            m = best_x.copy()
            # Sigma
            # Estimate initial sigma from the spread of initial points or use a heuristic
            # Heuristic: sigma = 0.5 * max_range / n or similar
            # Let's use a moderate initial sigma
            sigma = 1.0
            
            # Covariance matrix
            C = np.eye(n)
            
            # Step size control
            cs = (lambda_pop + 2) / (n + lambda_pop + 5)
            ds = 1 + 2 * max(0, np.sqrt((lambda_pop - 2) / (n + 2)) - 1) + cs
            hsig = (1.4 + 2 / (n + 1)) * (1 - np.exp(-1 / (ds * max_generations)))
            
            # Pre-compute weights
            weights = np.log(lambda_pop / 2) - np.log(np.arange(1, lambda_pop + 1))
            weights /= np.sum(weights)
            mu = int(np.sum(weights > 0))
            w_mu = weights[:mu]
            w_mu /= np.sum(w_mu)
            mu_eff = 1 / np.sum(w_mu ** 2)
            
            # Pre-compute matrix for C update
            # We need to generate N samples from N(0, C)
            # For efficiency, we can pre-compute the Cholesky factor if C is constant, 
            # but C changes. In 5D, computing Cholesky is cheap.
            
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            current_calls = len(xs)
            
            for gen in range(max_generations):
                if current_calls >= self.budget or time.time() > deadline:
                    break
                
                # Generate offspring
                offspring = []
                offspring_f = []
                
                # We need to generate lambda_pop points
                # x_i = m + sigma * B * z_i
                # Where B is the square root of C (Cholesky)
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback if C is not positive definite
                    B = np.eye(n)
                    
                for i in range(lambda_pop):
                    if current_calls >= self.budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    x = m + sigma * (B @ z)
                    x = np.clip(x, -5, 5)
                    f = func(x)
                    current_calls += 1
                    offspring.append(x)
                    offspring_f.append(f)
                
                if not offspring:
                    break
                    
                # Sort offspring
                offspring_f = np.array(offspring_f)
                offspring = np.array(offspring)
                idx = np.argsort(offspring_f)
                offspring = offspring[idx]
                offspring_f = offspring_f[idx]
                
                # Update best
                if offspring_f[0] < best_f:
                    best_f = offspring_f[0]
                    best_x = offspring[0].copy()
                
                # Update mean
                m_new = m + sigma * (np.sum(w_mu * offspring[:mu], axis=1))
                
                # Update step size control
                z_sigma = (B.T @ (m_new - m)) / sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * z_sigma
                chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
                
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * chi_n:
                    sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update covariance
                hsig = (1.4 + 2 / (n + 1)) * (1 - np.exp(-1 / (ds * (gen + 1))))
                p_c = (1 - cs) * p_c + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * (m_new - m) / sigma
                
                c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
                c_sigma = (2 + 2 / mu_eff) / (n + 2 + 2 / mu_eff)
                
                C = (1 + c_c * c_sigma - c_c) * C + \
                    c_c * (np.outer(p_c, p_c) + (1 - (1 - c_sigma) ** 2) * C) - \
                    c_sigma * (1 - (1 - c_sigma) ** 2) * C + \
                    c_sigma * np.sum([w_mu[i] * np.outer(offspring[i] - m, offspring[i] - m) for i in range(mu)], axis=0)
                
                m = m_new
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Regularize if needed
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.any(eigvals < 1e-12):
                    C = eigvecs @ np.diag(np.maximum(eigvals, 1e-12)) @ eigvecs.T

        # 3. Final Polish (Optional, if time and budget remain)
        # Use Nelder-Mead on the best point found
        if current_calls < self.budget and time.time() < deadline - 5:
            try:
                from scipy.optimize import minimize
                # Define a wrapper to count calls
                calls_made = [0]
                def wrapped_func(x):
                    calls_made[0] += 1
                    if calls_made[0] > self.budget - current_calls:
                        return 1e10
                    return func(x)
                
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8})
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
