"""Direct optimizer exploiting Rastrigin periodicity via multi-start CMA-ES."""
import numpy as np
import time
import math


class Optimizer:
    """Direct optimizer using multi-start CMA-ES with restarts for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline and calls_used < budget

        # Phase 1: Initial random sampling
        n_random = 20
        for _ in range(n_random):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        if not time_ok():
            return (best_x, best_f)

        # Phase 2: Multi-start CMA-ES
        # We will run CMA-ES starting from the best point found so far.
        # If it stagnates or uses up its allocated budget, we restart from a new random point
        # or the next best point from the initial sampling.

        # Standard CMA-ES implementation
        def run_cmaes(x0, max_evals):
            nonlocal calls_used, best_x, best_f
            
            # CMA-ES parameters
            lambda_ = 4 + int(3 * math.log(dim))  # population size
            mu = lambda_ // 2  # number of weights
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights ** 2)
            
            # Learning rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = (2 - c_1_1 + mu_eff / (dim + 2)) / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            
            # Initial values
            x_mean = x0.copy()
            sigma = 0.5  # Initial step size
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            
            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            evals_used = 0
            stagnation_counter = 0
            last_best_f = best_f
            
            while evals_used < max_evals and time_ok():
                # Generate population
                offspring = []
                for i in range(lambda_):
                    if not time_ok() or evals_used >= max_evals:
                        break
                    z = rng.standard_normal(dim)
                    x_child = x_mean + sigma * (C_sqrt @ z)
                    x_child = np.clip(x_child, -5, 5)
                    f_child = evaluate(x_child)
                    evals_used += 1
                    offspring.append((f_child, x_child))
                
                if len(offspring) < lambda_:
                    break
                    
                # Sort offspring
                offspring.sort(key=lambda t: t[0])
                
                # Update best
                if offspring[0][0] < last_best_f - 1e-10:
                    last_best_f = offspring[0][0]
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    
                # Stagnation check: if no improvement for many generations, break
                if stagnation_counter > 10:
                    break
                
                # Update mean
                x_mean_new = np.zeros(dim)
                for i in range(mu):
                    x_mean_new += weights[i] * offspring[i][1]
                
                # Update sigma
                # Check if sigma is too small
                if sigma < 1e-10:
                    break
                
                # Update evolution path sigma
                p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt.T @ (x_mean_new - x_mean) / sigma)
                
                # Update h_sigma (damping)
                h_sigma = (np.linalg.norm(p_sigma) / math.sqrt(1 - (1 - c_sigma) ** (2 * (evals_used + 1)))) < (1.4 + 2 / (dim + 1)) * chi_N
                chi_N = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
                
                # Update sigma
                sigma *= math.exp(c_sigma / chi_N * (np.linalg.norm(p_sigma) / chi_N - 1))
                sigma = float(np.clip(sigma, 1e-10, 10.0))
                
                # Update evolution path covariance
                p_c = (1 - c_c) * p_c + h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean) / sigma
                
                # Update covariance matrix
                # Use Cholesky decomposition for stability
                try:
                    C_sqrt, _ = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = (C + C.T) / 2
                    C_sqrt, _ = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                    
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                
                # Rank-one update
                rank_one = np.outer(x_mean_new - x_mean, x_mean_new - x_mean)
                
                # Rank-mu update
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = (x_mean_new - x_mean) # This is not quite right for rank-mu, need individual steps
                    # Actually, standard CMA-ES rank-mu uses the weighted sum of outer products of the steps
                    # Let's simplify: use only rank-one if implementation is complex, but rank-mu is important.
                    # For simplicity and robustness in this short script, I will use a simplified update or standard library if possible.
                    # Since I can't rely on external CMA-ES library, I will stick to a robust 1+1 or (1+lambda) variant if full CMA-ES is too prone to bugs here.
                    # However, the inspiration suggests CMA-ES works well. Let's try to implement standard CMA-ES correctly.
                    
                # Re-implementing Rank-Mu correctly
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = (offspring[i][1] - x_mean) / sigma
                    rank_mu += weights[i] * np.outer(y_i, y_i)
                    
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * rank_mu
                C = (C + C.T) / 2
                
                # Update Cholesky
                try:
                    C_sqrt, _ = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If Cholesky fails, reset C to identity or add jitter
                    C = (C + C.T) / 2
                    C_sqrt, _ = np.linalg.cholesky(C + 1e-6 * np.eye(dim))
                    
                x_mean = x_mean_new
                
                # Convergence check
                if sigma * np.linalg.norm(C_sqrt, ord=2) < 1e-8:
                    break

        # Allocate budget for CMA-ES runs
        remaining = budget - calls_used
        if remaining <= 0:
            return (best_x, best_f)
            
        # Split remaining budget into chunks
        # First run gets 50%, subsequent runs get 25%, 12.5%, etc.
        # Or just run until budget is low, then restart if stagnation occurs.
        
        # Simple strategy: Run CMA-ES with a large chunk, if it finishes early (stagnation), restart from best.
        # To ensure exploration, we also keep the initial best points.
        
        # Get top 5 points from initial sampling to use as restart seeds
        # We don't store all initial points in a list easily in the current structure, 
        # so we will just use the global best and random perturbations for restarts.
        
        seeds = [best_x.copy()]
        # Add a few random points as alternative seeds
        for _ in range(4):
            seeds.append(rng.uniform(-5, 5, dim))
            
        seed_idx = 0
        while time_ok() and calls_used < budget - 5:
            if seed_idx >= len(seeds):
                # Generate new random seeds
                seeds.append(rng.uniform(-5, 5, dim))
                
            max_evals = min(remaining - (calls_used - (budget - remaining)), 100) # Rough calculation
            # Better: just allocate a fixed chunk, e.g., 100 evals, or remaining if small.
            chunk = min(100, budget - calls_used - 5)
            if chunk < 10:
                break
                
            run_cmaes(seeds[seed_idx], chunk)
            
            # Update seeds for next restart: use current best
            seeds[0] = best_x.copy()
            seed_idx = (seed_idx + 1) % len(seeds)
            
            # If we used up most of the budget, break
            if budget - calls_used < 10:
                break

        return (best_x, best_f)
