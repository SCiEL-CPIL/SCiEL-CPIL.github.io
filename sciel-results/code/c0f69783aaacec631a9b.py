import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        
        global_best_x = None
        global_best_f = np.inf
        
        rng = np.random.default_rng(42)
        
        lb = -5.0
        ub = 5.0
        
        def evaluate(x):
            nonlocal calls_made, global_best_x, global_best_f
            fx = func(x)
            calls_made += 1
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()
            return fx

        def cma_es(x0, sigma0):
            nonlocal calls_made, global_best_x, global_best_f
            
            # State variables
            mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # CMA-ES parameters for dim=5
            lam = 9
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            # Initial population
            pop = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                x = mean + sigma * z
                x = np.clip(x, lb, ub)
                pop.append((x, evaluate(x)))
                if calls_made >= total_budget:
                    return
            
            while calls_made < total_budget and time.time() < deadline - 0.5:
                # Sort population by fitness
                pop.sort(key=lambda t: t[1])
                
                # Update mean
                new_mean = np.sum([weights[i] * pop[i][0] for i in range(mu)], axis=0)
                
                # Update step size
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * C @ (new_mean - mean)
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim):
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
                    
                # Update covariance
                diff = (new_mean - mean) / sigma
                pc = (1 - c_c) * pc + c_c * diff
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (c_c * (2 - c_c) - 1) * C)
                for i in range(mu):
                    x_i = (pop[i][0] - mean) / sigma
                    C += c_mu * weights[i] * np.outer(x_i, x_i)
                
                mean = new_mean
                sigma = max(sigma, 1e-10)
                
                # Check for convergence
                if sigma < 1e-6:
                    break
                
                # Generate next population
                next_pop = []
                for i in range(lam):
                    if calls_made >= total_budget:
                        break
                    z = rng.standard_normal(dim)
                    x = mean + sigma * z
                    x = np.clip(x, lb, ub)
                    fx = evaluate(x)
                    next_pop.append((x, fx))
                
                if len(next_pop) < lam:
                    break
                pop = next_pop

        # Strategy: Multiple restarts of CMA-ES with different initializations
        # Use a larger budget for the first restart to find a good basin
        # Then use remaining budget for additional restarts
        
        # First restart: larger initial sigma to explore broadly
        x0 = rng.uniform(-5, 5, dim)
        sigma0 = 2.0
        cma_es(x0, sigma0)
        
        # Additional restarts with smaller sigma, centered near global best or random
        for restart in range(2):
            if calls_made >= total_budget or time.time() > deadline - 1.0:
                break
            
            if global_best_x is not None and rng.random() < 0.6:
                x0 = global_best_x + 1.5 * rng.standard_normal(dim)
            else:
                x0 = rng.uniform(-5, 5, dim)
            x0 = np.clip(x0, -5, 5)
            sigma0 = 1.0
            
            cma_es(x0, sigma0)
        
        # Final polish with Nelder-Mead if budget remains
        if global_best_x is not None and calls_made < total_budget - 10 and time.time() < deadline - 0.5:
            try:
                from scipy.optimize import minimize
                res = minimize(lambda x: func(x), global_best_x, method='Nelder-Mead', 
                               options={'maxiter': 30, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < global_best_f:
                    global_best_f = res.fun
                    global_best_x = res.x
            except Exception:
                pass

        if global_best_x is None:
            global_best_x = np.zeros(dim)
            global_best_f = func(global_best_x)
            calls_made += 1

        return global_best_x, global_best_f
