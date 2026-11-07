import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = 0

    def __call__(self, func):
        start_time = time.time()
        self.deadline = start_time + 270  # Safety margin

        def evaluate(x):
            x = np.clip(x, -5, 5)
            f = func(x)
            self.calls += 1
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        def cma_es_run(func, mean, sigma, budget, dim, pop_size):
            if pop_size < 2:
                pop_size = 4
            mu = max(1, pop_size // 2)
            
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            mean_prev = mean.copy()
            best_f_run = np.inf
            best_x_run = mean.copy()
            calls_in_run = 0
            
            while calls_in_run < budget and self.calls < self.budget and time.time() < self.deadline:
                z = np.random.randn(pop_size, dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)
                    
                pop = mean + sigma * (z @ L.T)
                
                f_vals = []
                for i in range(pop_size):
                    if calls_in_run >= budget or self.calls >= self.budget or time.time() >= self.deadline:
                        break
                    f_i = evaluate(pop[i])
                    calls_in_run += 1
                    f_vals.append((f_i, pop[i].copy()))
                    
                    if f_i < best_f_run:
                        best_f_run = f_i
                        best_x_run = pop[i].copy()
                
                if len(f_vals) == 0:
                    break
                    
                f_vals.sort(key=lambda x: x[0])
                
                new_mean = np.zeros(dim)
                for i in range(min(mu, len(f_vals))):
                    new_mean += weights[i] * f_vals[i][1]
                mean = new_mean
                
                try:
                    L_inv = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    L_inv = np.eye(dim)
                    
                step = L_inv @ (mean - mean_prev)
                
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (step / sigma)
                
                norm_p_sigma = np.linalg.norm(p_sigma)
                denom = np.sqrt(1 - (1 - c_sigma)**(2 * (calls_in_run + 1)))
                if denom == 0:
                    h_sigma = 0
                else:
                    h_sigma = 1 if (norm_p_sigma / denom < (1.4 + 2 / (dim + 1)) * chi_n) else 0
                    
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (step / sigma)
                
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
                for i in range(min(mu, len(f_vals))):
                    y_i = L_inv @ (f_vals[i][1] - mean_prev)
                    C += c_mu * weights[i] * np.outer(y_i, y_i)
                    
                sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
                
                mean_prev = mean.copy()
                
                if sigma < 1e-12:
                    break
                    
            return best_f_run, best_x_run

        # IPOP-CMA-ES Style Strategy
        # Sequential restarts with increasing population size and sigma
        # This helps escape local minima by widening the search space in later stages
        
        best_f_global = np.inf
        best_x_global = np.zeros(self.dim)
        
        # Configuration: (population_size, budget_allocation, sigma)
        # We divide the budget into 4 phases.
        # Phase 1: Small pop, small sigma (local refinement)
        # Phase 2: Med pop, med sigma (escape local)
        # Phase 3: Large pop, large sigma (global exploration)
        # Phase 4: Largest pop, largest sigma (final global search)
        
        total_budget = self.budget
        # Allocate 20%, 20%, 30%, 30%
        b1 = int(total_budget * 0.2)
        b2 = int(total_budget * 0.2)
        b3 = int(total_budget * 0.3)
        b4 = total_budget - b1 - b2 - b3
        
        phases = [
            (6, b1, 1.0),
            (10, b2, 1.5),
            (14, b3, 2.0),
            (20, b4, 3.0)
        ]
        
        for pop, budget, sigma in phases:
            if self.calls >= self.budget or time.time() >= self.deadline:
                break
            
            # Start from the current global best
            # Resetting C and sigma allows for a fresh start with wider search
            f, x = cma_es_run(func, best_x_global, sigma, budget, self.dim, pop)
            
            if f < best_f_global:
                best_f_global = f
                best_x_global = x
        
        return self.best_x, self.best_f

    def _calc_weights(self, mu):
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        return weights
