"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """CMA-ES with multi-start for BBOB f5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        budget = self.budget
        
        # Track global best
        best_x = None
        best_f = np.inf
        
        # CMA-ES parameters
        # For dim=5, lambda = 4 + floor(3 * log(5)) = 4 + 3*1.6 = 8. Let's use 10 for slightly better exploration.
        lambda_ = 10
        # Standard initial step size
        sigma = 2.0
        
        # We will perform multiple restarts.
        # Each restart gets a portion of the budget.
        # Let's split the budget into chunks.
        # A single CMA-ES run usually converges in ~100-200 evaluations for dim=5 if well-initialized.
        # With 500 budget, we can do 2-3 restarts.
        
        # Strategy: 
        # 1. First restart: Center (0,0,0,0,0), sigma=2.0, budget=300
        # 2. Second restart: Random point, sigma=1.0, budget=100
        # 3. Third restart: Random point, sigma=0.5, budget=100
        
        # However, we must be careful with time.
        # CMA-ES is fast, so we are likely time-safe if we cap iterations.
        
        def run_cmaes(x0, sigma_init, budget_limit):
            nonlocal best_x, best_f, budget
            nonlocal deadline
            
            if budget < budget_limit:
                budget_limit = budget
                
            if budget_limit < 2 * lambda_ + dim:
                return
                
            # Initialize
            x_mean = np.array(x0, dtype=float)
            sigma = sigma_init
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Precompute constants
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))
            
            # Evaluation counter for this run
            calls_made = 0
            
            # Initial population
            # We need to generate lambda points.
            # To save time and ensure we don't exceed budget, we check budget before each eval.
            
            while calls_made < budget_limit:
                if time.time() > deadline:
                    return
                
                # Generate offspring
                z = np.random.randn(dim, lambda_)
                B = np.linalg.cholesky(C)
                # y = B * z
                y = (B @ z.T).T
                x_pop = x_mean + sigma * y
                
                # Evaluate
                f_pop = np.zeros(lambda_)
                for i in range(lambda_):
                    if calls_made >= budget_limit:
                        break
                    if time.time() > deadline:
                        return
                    # Clip to box [-5, 5]
                    x_eval = np.clip(x_pop[i], -5, 5)
                    f_pop[i] = func(x_eval)
                    calls_made += 1
                    budget -= 1
                    
                    if f_pop[i] < best_f:
                        best_f = f_pop[i]
                        best_x = x_eval.copy()
                
                if calls_made >= budget_limit:
                    break
                    
                # Sort
                idx = np.argsort(f_pop)
                y = y[idx]
                f_pop = f_pop[idx]
                
                # Update mean
                x_mean_new = x_mean + sigma * np.sum(weights[:, None] * y[:mu], axis=0)
                
                # Check for stagnation or step size adaptation
                # Calculate p_sigma
                y_sigma = y[:mu]
                # We need B and C inverse sqrt? 
                # Standard CMA-ES uses:
                # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2-c_sigma) mu_eff) / ||B|| * B^T C^{-1/2} w^T y
                # This is expensive to compute exactly without precomputing inverse roots.
                # For dim=5, we can approximate or use a simplified update.
                # Let's use a simplified update for speed and stability in this small dim.
                # Actually, let's just update the mean and covariance, and adapt sigma based on the norm of the step.
                
                # Simple Sigma adaptation (1/5 rule style or similar)
                # If the step was successful, increase sigma, else decrease.
                # A robust heuristic for small budgets:
                # If best_f improved significantly, keep sigma or increase slightly.
                # Else decrease.
                
                # Let's implement a basic covariance update.
                # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - (1-c_c)^2) delta(h) C) + c_mu sum w_i (y_i y_i^T)
                
                # To avoid heavy linear algebra in the loop which might be slow in Python for many iterations,
                # and since dim=5 is small, we can use a diagonal covariance approximation (Sep-CMA) 
                # or just standard CMA with numpy operations.
                
                # Given the 300s limit, numpy linalg on 5x5 is fast.
                
                # Update p_c
                h_sigma = 0 # Simplified, usually checks if p_sigma is too large
                p_c = (1 - c_c) * p_c + c_c * (x_mean_new - x_mean) / sigma
                
                # Update C
                # We need the weighted sum of y y^T
                y_w = weights[:mu]
                # Outer products
                # y_mu = y[:mu]
                # C_update = c_1 * (p_c/sigma) outer (p_c/sigma) + c_mu * sum(y_w_i * y_i outer y_i)
                
                pc_vec = p_c / sigma
                C_rank1 = np.outer(pc_vec, pc_vec)
                
                # Sum of weighted outer products
                # y is (mu, dim)
                # weights are (mu,)
                # We want sum_{i=1}^mu w_i y_i y_i^T
                # This is (y * w) @ y.T
                y_weighted = y[:mu] * y_w[:, None]
                C_rankmu = y_weighted @ y[:mu].T
                
                C_new = (1 - c_1 - c_mu) * C + c_1 * C_rank1 + c_mu * C_rankmu
                
                # Ensure symmetry and positive semi-definite
                C_new = (C_new + C_new.T) / 2.0
                # Eigenvalue check? For dim 5, we can just project if needed, but usually stable.
                C = C_new
                
                # Update mean
                x_mean = x_mean_new
                
                # Adapt sigma
                # Simple rule: if the norm of the step is larger than expected, reduce sigma.
                # Expected norm is sqrt(dim)
                step_norm = np.linalg.norm(x_mean_new - (x_mean_new - (x_mean_new - x_mean))) # This is just sigma * norm(y_mean)
                # Better:
                y_mean = np.sum(weights[:, None] * y[:mu], axis=0)
                step_norm = sigma * np.linalg.norm(y_mean)
                target_norm = math.sqrt(dim)
                
                if step_norm > target_norm * 1.5:
                    sigma *= 0.8
                elif step_norm < target_norm * 0.5:
                    sigma *= 1.2
                    if sigma > 10: sigma = 10 # Cap
                else:
                    # Slight decay to encourage convergence
                    sigma *= 0.995
                    
                if sigma < 1e-6:
                    break

        # Restart 1: Center
        x0_1 = np.zeros(dim)
        run_cmaes(x0_1, 2.0, 300)
        
        # Restart 2: Random point 1
        if time.time() < deadline and budget > 50:
            rng = np.random.default_rng(1)
            x0_2 = rng.uniform(-3, 3, dim)
            run_cmaes(x0_2, 1.0, 100)
            
        # Restart 3: Random point 2
        if time.time() < deadline and budget > 50:
            rng = np.random.default_rng(2)
            x0_3 = rng.uniform(-3, 3, dim)
            run_cmaes(x0_3, 0.5, 100)
            
        # If we still have budget, do a final random search?
        # Or just return best.
        
        if best_x is None:
            # Fallback if no calls made
            best_x = np.zeros(dim)
            best_f = func(best_x)
            
        return best_x, best_f

# EVOLVE-BLOCK-END
