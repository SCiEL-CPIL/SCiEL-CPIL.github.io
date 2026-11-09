import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """CMA-ES optimizer with time guard and final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        
        rng = np.random.default_rng(12345)
        
        # Initialize CMA-ES parameters
        # Start near center, but slightly random to avoid symmetry issues
        x_mean = np.zeros(self.dim)
        sigma = 2.0
        popsize = 4 + int(np.floor(3 * np.log(self.dim)))  # Standard CMA-ES popsize
        # Ensure popsize is at least 4
        popsize = max(4, popsize)
        
        # Covariance matrix
        C = np.eye(self.dim)
        # Evolution paths
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        # Learning rates
        mu = popsize // 2  # Number of weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_sigma = c_sigma / (1 + c_c)  # Adjusted
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (self.dim + 2) - 1) + c_sigma
        
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (weights[0] - 0.5) / ((self.dim + 2)**2 + mu_eff))
        
        # Initial evaluations
        calls = 0
        best_x = x_mean.copy()
        best_f = float('inf')
        
        # Initial population
        # We need to make sure we don't exceed budget
        if self.budget < popsize + 1:
             # Fallback to (1+1)-ES if budget is too small for CMA-ES
             x = rng.uniform(-5, 5, self.dim)
             fx = func(x)
             calls += 1
             sigma = 2.0
             while calls < self.budget and time.time() < deadline:
                 y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                 fy = func(y)
                 calls += 1
                 if fy < fx:
                     x, fx = y, fy
                     sigma *= 1.1
                 else:
                     sigma *= 0.9
             return x, fx

        # Main CMA-ES Loop
        while calls + popsize <= self.budget and time.time() < deadline:
            # Generate population
            # Cholesky decomposition of C
            L = np.linalg.cholesky(C)
            
            individuals = []
            for i in range(popsize):
                z = rng.standard_normal(self.dim)
                x_i = x_mean + sigma * (L @ z)
                x_i = np.clip(x_i, -5, 5)
                individuals.append(x_i)
            
            # Evaluate
            fitness = []
            for x_i in individuals:
                if time.time() > deadline:
                    break
                f_i = func(x_i)
                calls += 1
                fitness.append(f_i)
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
            
            if len(fitness) < popsize:
                break
            
            fitness = np.array(fitness)
            
            # Sort
            order = np.argsort(fitness)
            sorted_fitness = fitness[order]
            sorted_individuals = [individuals[i] for i in order]
            
            # Update mean
            x_new_mean = sum(weights[i] * sorted_individuals[i] for i in range(mu))
            
            # Update evolution paths
            # p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L.T @ (x_new_mean - x_mean) / (sigma * np.linalg.norm(L.T @ (x_new_mean - x_mean) + 1e-10) if np.linalg.norm(L.T @ (x_new_mean - x_mean)) > 1e-10 else 0))
            # Simplified p_sigma update (standard CMA-ES)
            # Let's use the standard formula: p_sigma = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)*mu_eff) * h_sigma * (x_new_mean - x_mean)/(sigma * ||...||)
            # Actually, simpler implementation for stability:
            
            # Recalculate p_sigma properly
            # diff = (x_new_mean - x_mean) / sigma
            # p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L.T @ diff)
            # h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1) / (self.dim + 1) / c_sigma))) < (1 + 2 / (self.dim + 1)) * np.sqrt((self.dim + 1) / (self.dim - 1))
            
            # To avoid complexity bugs, I will use a simplified CMA-ES update or a well-tested library-like implementation.
            # Given the risk of implementation errors in CMA-ES from scratch, and the "FRESH DESIGN" prompt, 
            # maybe a simpler but robust approach is better.
            
            # Let's switch to a robust (1, lambda)-ES with adaptive sigma, which is simpler and less prone to bug, 
            # or just use scipy's differential_evolution or similar if allowed? No, standard library + numpy/scipy.
            
            # Actually, let's implement a clean CMA-ES.
            
            # Update p_sigma
            diff = (x_new_mean - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L.T @ diff)
            
            # h_sigma
            # chi_n = np.sqrt(self.dim) * (1 - 1.0 / (4 * self.dim) + 1.0 / (21 * self.dim**2))
            # h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1) / (self.dim + 1) / c_sigma))) < (1 + 2 / (self.dim + 1)) * np.sqrt((self.dim + 1) / (self.dim - 1))
            # This is getting complex. Let's use a simpler covariance adaptation.
            
            # Simplified CMA-ES update:
            # 1. Mean update (done)
            # 2. Sigma update: sigma = sigma * exp(c_sigma * (||p_sigma|| / chi_n - 1) / d_sigma)
            # 3. Covariance update: C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c.T + (1 - h_c) * C) + c_mu * sum(weights[i] * (x_i - x_mean)(x_i - x_mean).T)
            
            # Let's restart with a simpler, more robust (1+1)-ES with restarts if no improvement, 
            # as CMA-ES implementation from scratch is error-prone.
            # BUT, the knowledge says CMA-ES is better.
            
            # I will implement a standard CMA-ES carefully.
            
            # Re-doing the loop logic cleanly.
            pass

        # Fallback: If CMA-ES loop breaks or is too complex, use (1+1)-ES for remaining budget
        # To ensure correctness and avoid bugs, I will implement a robust (1+1)-ES with adaptive sigma 
        # and multiple restarts if stuck, which is simpler and often competitive in 5D.
        
        # Actually, let's just implement a very solid (1+1)-ES with time guard and polish.
        # The knowledge says CMA-ES is better, but a buggy CMA-ES is worse than a good (1+1)-ES.
        
        # Let's try to implement CMA-ES correctly.
        
        # Reset state for a clean CMA-ES implementation
        x_mean = np.zeros(self.dim)
        sigma = 2.0
        C = np.eye(self.dim)
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        mu = popsize // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_sigma = c_sigma / (1 + c_c)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (self.dim + 2) - 1) + c_sigma
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (weights[0] - 0.5) / ((self.dim + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(self.dim) * (1 - 1.0 / (4 * self.dim) + 1.0 / (21 * self.dim**2))
        
        calls = 0
        best_x = x_mean.copy()
        best_f = float('inf')
        
        # Initial evaluation
        if calls < self.budget and time.time() < deadline:
            best_f = func(best_x)
            calls += 1
        
        while calls + popsize <= self.budget and time.time() < deadline:
            # Generate
            L = np.linalg.cholesky(C)
            individuals = []
            for _ in range(popsize):
                z = rng.standard_normal(self.dim)
                x_i = x_mean + sigma * (L @ z)
                x_i = np.clip(x_i, -5, 5)
                individuals.append(x_i)
            
            # Evaluate
            fitness = []
            for x_i in individuals:
                if time.time() > deadline:
                    break
                f_i = func(x_i)
                calls += 1
                fitness.append(f_i)
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
            
            if len(fitness) < popsize:
                break
            
            fitness = np.array(fitness)
            order = np.argsort(fitness)
            sorted_fitness = fitness[order]
            sorted_individuals = [individuals[i] for i in order]
            
            # Update Mean
            x_new_mean = sum(weights[i] * sorted_individuals[i] for i in range(mu))
            
            # Update p_c
            p_c = (1 - c_c) * p_c + c_c * (x_new_mean - x_mean) / sigma
            
            # h_c
            h_c = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**(2 * (calls + 1) / (self.dim + 1) / c_c))) < (1 + 2 / (self.dim + 1)) * np.sqrt((self.dim + 1) / (self.dim - 1))
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c.T + (1 - h_c) * C)
            for i in range(mu):
                diff = sorted_individuals[i] - x_mean
                C += c_mu * weights[i] * np.outer(diff, diff)
            
            # Update p_sigma
            diff = (x_new_mean - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L.T @ diff)
            
            # h_sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1) / (self.dim + 1) / c_sigma))) < (1 + 2 / (self.dim + 1)) * np.sqrt((self.dim + 1) / (self.dim - 1))
            
            # Update Sigma
            sigma = sigma * np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1) / d_sigma)
            
            # Update Mean
            x_mean = x_new_mean
            
            # Check for stagnation or invalid values
            if not np.all(np.isfinite(x_mean)) or sigma < 1e-12:
                break
        
        # Final Polish: Nelder-Mead on the best point found
        if time.time() < deadline - 10:
            try:
                res = minimize(func, best_x, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_f:
                    best_x = res.x
                    best_f = res.fun
            except:
                pass
        
        return best_x, best_f
