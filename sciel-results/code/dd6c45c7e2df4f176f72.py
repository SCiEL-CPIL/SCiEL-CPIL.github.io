import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290  # Safety margin before 300s limit
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        rng = np.random.default_rng(42)
        
        def try_eval(x):
            nonlocal best_f, best_x, eval_count
            if eval_count >= self.budget or time.time() > deadline - 2:
                return None
            try:
                fx = func(x)
            except Exception:
                return None
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial random sampling
        n_init = min(20, self.budget // 5)
        for i in range(n_init):
            x = rng.uniform(-5, 5, self.dim)
            try_eval(x)
        
        # Phase 2: CMA-ES with adaptive population size and restarts
        remaining = self.budget - eval_count
        if remaining > 0 and time.time() < deadline - 5:
            # Use multiple CMA-ES restarts with different initializations
            n_restarts = 3
            budget_per_restart = max(50, remaining // n_restarts)
            
            for restart in range(n_restarts):
                if eval_count >= self.budget or time.time() > deadline - 5:
                    break
                
                # Initialize CMA-ES parameters
                if restart == 0:
                    x0 = best_x.copy()
                    sigma0 = 1.0
                else:
                    # Different starting points for subsequent restarts
                    x0 = rng.uniform(-3, 3, self.dim)
                    try_eval(x0)
                    if eval_count >= self.budget or time.time() > deadline - 5:
                        break
                    sigma0 = 0.5
                
                # CMA-ES parameters
                lam = max(4, int(4 + 3 * math.log(self.dim)))  # population size
                c1 = (4 + lam / self.dim) / ((self.dim + 2)**2 + 4)
                cmu = min(0.1, (2 * (2 + 2.0 / (self.dim + 1.7)) - 1) / ((self.dim + 2)**2 + 4))
                cs = (lam + 2 + 2 / (lam + 2)) / (self.dim + lam + 5)
                c1 = (4 + lam / self.dim) / ((self.dim + 2)**2 + 4)
                cmu = min(0.1, (2 * (2 + 2.0 / (self.dim + 1.7)) - 1) / ((self.dim + 2)**2 + 4))
                damps = (self.dim + 1 + 2 * min(1, 1 / (lam - self.dim - 1))) / (self.dim + 4 + 2 / (lam - self.dim - 1))
                cc = (4 + self.dim / lam) / (self.dim + 4 + 4 / lam)
                
                # State variables
                mean = x0.copy()
                sigma = sigma0
                C = np.eye(self.dim)
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                
                # Precompute
                chi_n = math.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
                
                max_iter = max(10, (budget_per_restart - 1) // lam)
                
                for iteration in range(max_iter):
                    if eval_count >= self.budget or time.time() > deadline - 5:
                        break
                    
                    # Generate population
                    population = []
                    fitness = []
                    
                    for i in range(lam):
                        if eval_count >= self.budget or time.time() > deadline - 5:
                            break
                        z = rng.standard_normal(self.dim)
                        # Whitening: C^(1/2) z
                        L = np.linalg.cholesky(C)
                        x = mean + sigma * L @ z
                        x = np.clip(x, -5, 5)
                        fx = try_eval(x)
                        if fx is None:
                            break
                        population.append(x)
                        fitness.append(fx)
                    
                    if len(population) < 2:
                        break
                    
                    # Sort by fitness
                    indices = np.argsort(fitness)
                    population = [population[i] for i in indices]
                    fitness = [fitness[i] for i in indices]
                    
                    # Update best
                    if fitness[0] < best_f:
                        best_f = fitness[0]
                        best_x = population[0].copy()
                    
                    # Update mean with weighted recombination
                    mu = lam // 2
                    weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
                    weights /= weights.sum()
                    mueff = 1 / np.sum(weights**2)
                    
                    new_mean = np.zeros(self.dim)
                    for i in range(mu):
                        new_mean += weights[i] * population[i]
                    mean = new_mean
                    
                    # Update evolution paths
                    old_mean = mean - sigma * L @ (np.linalg.inv(L) @ (mean - x0)) if False else mean
                    # Simplified: update paths
                    ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) / sigma * L @ (mean - old_mean) / sigma if False else (1 - cs) * ps
                    # Actually let's do it properly
                    diff = (mean - old_mean) / sigma
                    ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * diff
                    
                    hsig = (np.linalg.norm(ps) / math.sqrt(1 - (1 - cs)**2) / chi_n) < (1.4 + 2/(self.dim+1))
                    hsig = float(hsig)
                    
                    pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * diff
                    
                    # Update covariance
                    delta_hsig = (1 - hsig) * cc * (2 - cc)
                    rank_one = np.outer(pc, pc)
                    rank_mu = np.zeros((self.dim, self.dim))
                    for i in range(mu):
                        zi = (population[i] - old_mean) / sigma
                        rank_mu += weights[i] * np.outer(zi, zi)
                    
                    C = (1 + cs * delta_hsig * cc * (2 - cc) - cs * mueff - c1 - cmu) * C \
                        + cs * delta_hsig * np.outer(pc, pc) \
                        + c1 * rank_one + cmu * rank_mu
                    
                    # Ensure positive semi-definite
                    C = (C + C.T) / 2
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-10)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    
                    # Update sigma
                    sigma *= math.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                    sigma = np.clip(sigma, 1e-10, 10.0)
                    
                    # Convergence check
                    if sigma < 1e-8 or iteration > 200:
                        break
        
        # Phase 3: Final local polish
        remaining = self.budget - eval_count
        if remaining > 0 and time.time() < deadline - 2:
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            rng_local = np.random.default_rng(123)
            
            for _ in range(min(remaining, 50)):
                if time.time() > deadline - 2:
                    break
                y = np.clip(x + sigma * rng_local.standard_normal(self.dim), -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 0.9
                else:
                    sigma *= 0.9
                    if sigma < 1e-8:
                        break
                        
        return best_x, best_f
