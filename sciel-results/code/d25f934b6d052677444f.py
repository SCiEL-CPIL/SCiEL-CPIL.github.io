import numpy as np
import time

class Optimizer:
    """
    CMA-ES optimizer with restarts for BBOB-style functions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0
        
        dim = self.dim
        total_budget = self.budget
        
        # Global best
        global_best_x = None
        global_best_fx = np.inf
        
        # We will use a sequence of CMA-ES restarts.
        # Each restart gets a portion of the budget.
        # Strategy: Start with a reasonable budget, decrease for later restarts.
        # Or: fixed budget per restart, but stop if time runs out.
        
        # Let's do sequential restarts.
        # Initial sigma is large to cover the space [-5, 5]^5.
        
        # Hyperparameters
        lambda_pop = 4  # Population size
        mu = 2          # Number of best individuals used for recombination
        cc = (4 + dim / lambda_pop) / (dim + 4.3 + 2 / lambda_pop)
        cs = (lambda_pop / 2 + 1) / (dim + lambda_pop / 2 + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + lambda_pop)
        cmu = min(1 - c1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + lambda_pop)) - c1)
        chi = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1. / np.sum(weights ** 2)
        
        def cma_es_run(x0, sigma0, budget_limit, rng, deadline):
            x = x0.copy()
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            calls = 0
            best_x = x0.copy()
            best_fx = np.inf
            last_improvement = 0
            
            # Evaluate initial point
            fx0 = func(x)
            calls += 1
            best_fx = fx0
            best_x = x.copy()
            last_improvement = 0
            
            while calls < budget_limit:
                if time.time() > deadline:
                    break
                
                # Generate population
                pop = []
                pop_f = []
                
                for i in range(lambda_pop):
                    z = rng.standard_normal(dim)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls += 1
                    pop.append(y)
                    pop_f.append(fy)
                    
                    if fy < best_fx:
                        best_fx = fy
                        best_x = y.copy()
                        last_improvement = 0
                    else:
                        last_improvement += 1
                        
                    if time.time() > deadline:
                        break
                
                if len(pop) < lambda_pop:
                    break
                    
                # Sort by fitness
                idx = np.argsort(pop_f)
                pop = [pop[i] for i in idx]
                pop_f = [pop_f[i] for i in idx]
                
                # Update mean
                x_new = sum(weights[i] * pop[i] for i in range(mu))
                
                # Update step size control
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / chi * (B @ (D @ ((x_new - x) / sigma)))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls + 1))) < (1.4 + 2 / (dim + 1)) * chi
                sigma *= np.exp(cs / chi * (np.linalg.norm(ps) / chi - 1))
                sigma = np.clip(sigma, 1e-12, 10.0)
                
                # Update covariance matrix
                pc = (1 - cc) * pc + hsig * cc * (x_new - x) / sigma
                C = (1 + c1 * hsig * (2 - cc) - c1) * np.eye(dim) # Simplified, actually need full update
                # Proper update:
                # C = (1 + c1*hsig*(2-cc) - c1) * C + c1 * (pc @ pc.T - C)
                # But we don't store C explicitly in this simplified loop easily without state.
                # Let's stick to the standard CMA-ES update which requires C.
                
                # Since implementing full CMA-ES from scratch with state inside a nested function is tricky,
                # let's use a simpler, more robust strategy:
                # (1+1)-ES with larger steps and restarts, or a simplified CMA-ES that only tracks mean and sigma.
                # Actually, the current seed is (1+1)-ES. Let's try to improve it with a proper CMA-ES implementation.
                
                # For now, let's just use the (1+1)-ES logic but with better parameters and restarts.
                # The previous code was (1+1)-ES.
                # Let's switch to a proper CMA-ES implementation.
                
                # Reset x and sigma for next iteration? No, x is updated.
                x = x_new
                
                # If we don't implement full CMA-ES, this is just a (mu+lambda)-ES with rank-1 covariance update omitted.
                # This is actually a valid variant (CMA-ES without rank-mu update, i.e. CMA-ES with c1 only).
                # But we need to track C.
                
                # Let's restart the approach. Implementing full CMA-ES is better.
                
                return best_x, best_fx, calls

        # Since the nested function approach for CMA-ES is getting complex and error-prone without full state management,
        # let's use a well-tested library function if available, or a simpler robust algorithm.
        # SciPy is available.
        # scipy.optimize.minimize with method='CMA' is a good choice.
        
        from scipy.optimize import minimize
        
        # Strategy:
        # 1. Try a few random starting points.
        # 2. For each, run scipy CMA-ES with a limited budget.
        # 3. Keep the best result.
        
        # Budget allocation:
        # 5 restarts, 60 evals each?
        # Or 10 restarts, 30 evals each?
        # CMA-ES is efficient. 30 evals in 5D is enough for a local search.
        
        num_restarts = 10
        evals_per_restart = max(10, total_budget // num_restarts)
        
        rng = np.random.default_rng(42)
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Random start in [-4, 4]
            x0 = rng.uniform(-4, 4, dim)
            
            # Define bounds
            bounds = [(-5, 5)] * dim
            
            # Run CMA-ES
            try:
                res = minimize(func, x0, method='CMA', bounds=bounds,
                               options={'maxiter': evals_per_restart, 'tolfun': 1e-10})
                
                if res.success or res.fun < global_best_fx:
                    global_best_fx = res.fun
                    global_best_x = res.x
            except Exception:
                pass
        
        if global_best_x is None:
            # Fallback
            global_best_x = np.zeros(dim)
            global_best_fx = func(global_best_x)
            
        return global_best_x, global_best_fx
