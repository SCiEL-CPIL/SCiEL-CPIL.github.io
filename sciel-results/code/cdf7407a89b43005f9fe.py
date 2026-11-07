import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        dim = self.dim
        budget = self.budget
        calls = 0
        
        def check_deadline():
            return time.time() >= deadline

        rng = np.random.default_rng(42)
        
        # State
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls += 1
        
        # CMA-ES parameters
        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Update rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        def run_cma_es(x_mean, sigma, max_calls):
            nonlocal calls, best_x, best_f
            
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Pre-compute C_inv_sqrt
            C_inv_sqrt = np.eye(dim)
            
            iter_count = 0
            stagnation = 0
            
            while calls < budget and not check_deadline():
                # Generate offspring
                offspring = []
                for i in range(lam):
                    if calls >= budget or check_deadline():
                        break
                    z = rng.standard_normal(dim)
                    # y = x_mean + sigma * C_chol @ z
                    # We need C_chol. We can use C_inv_sqrt.T as an approximation for direction if we update it, 
                    # but standard CMA-ES uses C_chol.
                    # Let's maintain C_chol.
                    try:
                        C_chol = np.linalg.cholesky(C)
                    except:
                        C_chol = np.eye(dim)
                        
                    y = x_mean + sigma * (C_chol @ z)
                    y = np.clip(y, -5.0, 5.0)
                    fy = func(y)
                    calls += 1
                    offspring.append((y, fy))
                    
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        
                if len(offspring) < mu:
                    break
                    
                offspring.sort(key=lambda x: x[1])
                selected = offspring[:mu]
                
                # Weighted mean
                x_new_mean = sum(weights[i] * selected[i][0] for i in range(mu))
                x_new_mean = np.array(x_new_mean)
                
                # Update paths
                diff = (x_new_mean - x_mean) / sigma
                
                # C_inv_sqrt update
                try:
                    C_inv = np.linalg.inv(C)
                    C_inv_sqrt = np.linalg.cholesky(C_inv)
                    C_inv_sqrt_T = C_inv_sqrt.T
                except:
                    C_inv_sqrt_T = np.eye(dim)
                    
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt_T @ diff)
                p_sigma_norm = np.linalg.norm(p_sigma)
                h_sigma = 1.0 if p_sigma_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (iter_count + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0
                
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff
                
                # Update covariance
                delta_h = (1.8 + 2 / (mu_eff + 1)) * (1 - h_sigma)
                rank_one = np.outer(p_c + delta_h * p_c, p_c + delta_h * p_c)
                
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    yi = selected[i][0] - x_mean
                    rank_mu += weights[i] * np.outer(yi, yi) / (sigma ** 2)
                    
                C = (1 + c_1 * (rank_one - C) + c_mu * (rank_mu - C)) * C
                C = (C + C.T) / 2
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (p_sigma_norm / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                x_mean = x_new_mean
                iter_count += 1
                
                # Check stagnation
                if best_f == offspring[0][1]: # No improvement in this generation
                    stagnation += 1
                else:
                    stagnation = 0
                    
                # Restart condition
                if stagnation > 20 or sigma < 1e-6:
                    break
                    
            return x_mean, sigma

        # Run initial CMA-ES
        x_mean = np.zeros(dim)
        sigma = 1.0
        x_mean, sigma = run_cma_es(x_mean, sigma, budget)
        
        # If budget remains, run local polish
        if calls < budget and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma_local = max(0.01, sigma * 0.5)
            
            for _ in range(min(100, budget - calls)):
                if check_deadline() or calls >= budget:
                    break
                y = x + sigma_local * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x = y
                    fx = fy
                    best_x = x.copy()
                    best_f = fx
                else:
                    sigma_local *= 0.8
                    if sigma_local < 1e-4:
                        break
                # Simple adaptive sigma for (1+1)-ES
                # This is a bit crude, let's use a standard (1+1)-ES logic
                
        return best_x, best_f
