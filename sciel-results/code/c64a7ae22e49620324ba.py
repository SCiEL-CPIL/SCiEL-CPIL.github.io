import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        
        n = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            # Ensure x is a numpy array of correct shape
            x = np.asarray(x, dtype=np.float64)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def run_cmaes(start_mean, start_sigma, budget_limit, lambda_, mu):
            nonlocal calls_made, best_x, best_f
            
            if calls_made >= budget_limit:
                return
                
            # CMA-ES Parameters
            # Ensure mu < lambda
            if mu >= lambda_:
                mu = lambda_ // 2
                
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            
            # Adaptation parameters
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            
            # State
            mean = start_mean.copy()
            sigma = start_sigma
            C = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            gen = 0
            while calls_made < budget_limit:
                if time.time() > deadline:
                    break
                
                # Generate population
                z = np.random.standard_normal((lambda_, n))
                
                # Compute C_sqrt (Cholesky)
                try:
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                    C_sqrt = L.T
                except:
                    # Fallback: Eigendecomposition
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                    
                # Sample points
                # y = z @ C_sqrt.T
                y = np.einsum('ij,jk->ik', z, C_sqrt.T)
                x_pop = mean + sigma * y
                
                # Evaluate population
                f_pop = np.zeros(lambda_)
                for i in range(lambda_):
                    if calls_made >= budget_limit:
                        break
                    f_pop[i] = evaluate(x_pop[i])
                
                if calls_made >= budget_limit:
                    break
                    
                # Sort indices
                idx = np.argsort(f_pop)
                
                # Update mean
                x_w = np.zeros(n)
                for i in range(mu):
                    x_w += weights[i] * x_pop[idx[i]]
                
                # Update p_sigma
                # C_inv_sqrt
                try:
                    C_inv_sqrt = np.linalg.inv(C_sqrt)
                except:
                    C_inv_sqrt = np.eye(n)
                    
                y_w = (x_w - mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mueff) * (C_inv_sqrt @ y_w)
                
                # Update p_c
                # Standard h_sigma check
                norm_p_sigma = np.linalg.norm(p_sigma)
                # Approximate generations for the threshold
                # 1 - (1-c_sigma)^(2*gen) is close to 1 for large gen
                # For small gen, use the exact formula
                # gen is incremented below, so use gen+1 for current update? 
                # Standard CMA-ES uses the current generation count.
                # Let's use gen as the number of completed generations.
                # The threshold is sqrt(1 - (1-c_sigma)^(2*gen))
                # If gen=0, this is 0, which is problematic. 
                # Usually, the check is done after the first generation.
                if gen > 0:
                    threshold_denom = np.sqrt(1 - (1 - c_sigma)**(2 * gen))
                    if threshold_denom < 1e-10:
                        threshold_denom = 1e-10
                    h_sigma = (norm_p_sigma / threshold_denom) < (1.4 + 2 / (n + 1)) * np.sqrt((mueff + 1) / (mueff + 3))
                else:
                    h_sigma = 1.0
                
                p_c = (1 - c_c) * p_c + h_sigma * c_c * (x_w - mean) / sigma
                
                # Update Covariance C
                C_update = np.zeros((n, n))
                for i in range(mu):
                    y_i = (x_pop[idx[i]] - mean) / sigma
                    C_update += weights[i] * np.outer(y_i, y_i)
                    
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (1 - c_c) * C) + c_mu * C_update
                
                # Update mean
                mean = x_w
                
                # Update sigma
                sigma *= np.exp((c_sigma / d_sigma) * (norm_p_sigma / np.sqrt(mueff) - 1))
                
                # Ensure sigma is reasonable
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                gen += 1
                
                # Convergence check
                if sigma < 1e-8:
                    break

        # Phase 1: Global Search
        # Start at 0, sigma=4.0 to cover [-5, 5]^5
        # Reserve 50 calls for polish
        global_budget = total_budget - 50
        if calls_made < global_budget:
            initial_mean = np.zeros(n)
            # If we have a previous best, we might start there, but for global search, 0 is a good heuristic for f24
            run_cmaes(initial_mean, 4.0, global_budget, lambda_=8, mu=4)
            
        # Phase 2: Local Polish
        # Use a smaller sigma and smaller population for high precision
        if best_x is not None and calls_made < total_budget:
            # Start from best_x found in Phase 1
            # Use a small sigma, e.g., 0.1 or 0.2
            # Small population: lambda=4, mu=2
            run_cmaes(best_x, 0.2, total_budget, lambda_=4, mu=2)

        # Fallback if no calls were made (should not happen)
        if best_x is None:
            x = np.zeros(n)
            best_f = func(x)
            best_x = x

        return best_x, best_f
