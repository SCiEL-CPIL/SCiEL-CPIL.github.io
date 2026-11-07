"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Continuous optimizer for BBOB-style f9 (Rosenbrock rotated) on [-5,5]^5.
    
    Strategy:
    - Phase 1: CMA-ES-like global search with adaptive step size and covariance.
    - Phase 2: Local refinement with (1+1)-ES from the best point found.
    - Time guard: track wall-clock to stay within 330s limit with safety margin.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # 10s safety margin before 330s kill
        
        # Seeded RNG for reproducibility
        rng = np.random.default_rng(42)
        
        # Track best solution found
        best_x = np.zeros(n)
        best_f = np.inf
        calls_made = 0
        
        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            fx = func(np.asarray(x, dtype=np.float64))
            calls_made += 1
            return fx
        
        # Phase 1: CMA-ES-like search
        # Initialize with a diverse set of starting points
        n_restarts = 3
        best_overall_x = None
        best_overall_f = np.inf
        
        for restart in range(n_restarts):
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Initialize CMA-ES parameters
            if restart == 0:
                x = np.zeros(n)  # Start near origin
                sigma = 2.0
            elif restart == 1:
                x = rng.uniform(-3, 3, n)
                sigma = 1.5
            else:
                x = rng.uniform(-4, 4, n)
                sigma = 1.0
            
            # Initialize covariance matrix
            C = np.eye(n)
            
            # Evolution path
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            # Learning rates
            c_sigma = (1 + 2 / max(0, (n - 2))) / (n + 4)
            c_c = (4 + n / 4) / (n + 4)
            mu = max(1, n // 2)  # Number of selected individuals
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1 / (w @ w)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_sigma = (1 + 2 / max(0, (n - 2)) + mu_eff / (n + 4)) / (n + 4 + 3 * mu_eff / n)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            
            # Population size
            lambda_pop = 4 + int(3 * np.log(n))
            
            max_cmaes_iter = 150
            for it in range(max_cmaes_iter):
                if time.time() > deadline or calls_made >= budget:
                    break
                
                # Sample population
                offspring = []
                for j in range(lambda_pop):
                    z = rng.standard_normal(n)
                    y = np.clip(x + sigma * C @ z, -5, 5)
                    fy = safe_call(y)
                    if fy < best_overall_f:
                        best_overall_f = fy
                        best_overall_x = y.copy()
                    offspring.append((y, fy))
                
                if len(offspring) < lambda_pop:
                    break
                
                # Sort by fitness
                offspring.sort(key=lambda t: t[1])
                
                # Update mean
                y_w = np.array([offspring[j][0] for j in range(mu)])
                y_mean = y_w.T @ w
                x = x + y_mean
                
                # Update sigma
                y_sigma = y_w.T @ w
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C @ y_sigma)
                chi_n = np.sqrt(n) * (1 + 1 / (4 * n) - 1 / (21 * n**2))
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update covariance
                y_c = y_w.T @ w
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C @ y_c)
                C = (1 + c_c * (2 - c_c) * mu_eff - c_c) * C + c_c * (np.outer(y_c, y_c) / mu_eff - np.outer(p_c, p_c))
                C = (C + C.T) / 2  # Symmetrize
                
                # Prevent sigma from becoming too large or too small
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # Check convergence
                if np.linalg.norm(sigma * C @ rng.standard_normal(n)) < 1e-10:
                    break
            
            # After each restart, if we improved, continue with better initial point
            if best_overall_x is not None and calls_made < budget and time.time() < deadline:
                # Use the best point as a starting hint for next phase
                pass
        
        # Phase 2: Local refinement with (1+1)-ES
        if best_overall_x is not None and calls_made < budget and time.time() < deadline:
            x = best_overall_x.copy()
            fx = best_overall_f
            sigma = 0.5
            local_budget = min(100, budget - calls_made)
            
            for _ in range(local_budget):
                if time.time() > deadline or calls_made >= budget:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = safe_call(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fy < best_overall_f:
                        best_overall_f = fy
                        best_overall_x = x.copy()
                else:
                    sigma *= 0.85
                
                # If step size is too small, stop
                if sigma < 1e-6:
                    break
        
        # Return the best solution found
        if best_overall_x is None:
            # Fallback: return a reasonable default
            best_overall_x = np.zeros(n)
            best_overall_f = safe_call(best_overall_x)
        
        return best_overall_x, best_overall_f
# EVOLVE-BLOCK-END
