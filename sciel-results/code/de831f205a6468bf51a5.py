"""
Optimization strategy:
1. Initial random sampling to identify promising basins.
2. CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for local/global refinement.
CMA-ES is chosen for its robustness in navigating the complex landscape of the Rastrigin function,
adapting the search distribution shape to the local curvature.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def check_deadline():
            return time.time() >= deadline or calls_made >= budget

        # Phase 1: Exploratory Random Search
        # Sample points to find a good starting basin
        n_explore = min(30, budget // 10)
        explore_x = rng.uniform(-5, 5, (n_explore, dim))
        explore_f = np.array([evaluate(x) for x in explore_x])
        
        # Sort by fitness
        sort_idx = np.argsort(explore_f)
        best_start = explore_x[sort_idx[0]].copy()
        best_start_f = explore_f[sort_idx[0]]

        # Phase 2: CMA-ES Refinement
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # Population size
        if lambda_ < 4:
            lambda_ = 4
        
        # Initialize CMA-ES state
        mean = best_start.copy()
        sigma = 0.5  # Initial step size, tuned for Rastrigin basin width
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # CMA-ES update parameters
        mu = lambda_ // 2  # Number of selected points
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Run CMA-ES
        while not check_deadline():
            # Sample population
            # Whitened normal distribution
            z = rng.standard_normal((lambda_, dim))
            # Transform using Cholesky of C
            # Since C is updated via rank-1 and rank-mu updates, we maintain it as a matrix.
            # For efficiency, we can use the fact that C is symmetric positive definite.
            # We need the square root of C. np.linalg.cholesky works for SPD.
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes non-SPD due to numerical issues, reset to identity or diagonal
                C = np.eye(dim)
                B = np.eye(dim)
            
            pop = mean + sigma * (z @ B.T)
            
            # Evaluate population
            # Note: We must ensure we don't exceed budget in the middle of a generation
            # If we run out of budget mid-generation, we just stop.
            f_values = np.empty(lambda_)
            for i in range(lambda_):
                if check_deadline():
                    # If we run out, break. The loop condition will catch it next time.
                    # But we need to handle partial evaluation? 
                    # The task says "More than 500 calls... makes result invalid".
                    # So we must stop BEFORE exceeding 500.
                    # If we have < lambda_ calls left, we should break.
                    break
                f_values[i] = evaluate(pop[i])
            
            if calls_made >= budget or time.time() >= deadline:
                break
                
            # If we didn't evaluate all, resize
            if len(f_values) < lambda_:
                 # This shouldn't happen if we check before, but just in case
                 # We need to re-sort only the evaluated ones? 
                 # Actually, if we break early, we have fewer points. 
                 # Standard CMA-ES expects lambda points. 
                 # To be safe, if we can't complete a generation, we stop.
                 break

            # Sort by fitness
            order = np.argsort(f_values)
            y = pop[order]
            f_sorted = f_values[order]
            
            # Update mean
            mean_new = np.sum(weights[:, np.newaxis] * y[:mu], axis=0)
            
            # Update step-size control (CSA)
            ps = (1 - cs) * ps + cs * np.sqrt(mu_eff) / chiN * (B.T @ ((mean_new - mean) / sigma))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / chiN) < (1.4 + 2 / (dim + 1))
            
            # Update covariance matrix
            pc = (1 - cc) * pc + hsig * cc * (mean_new - mean) / sigma
            
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 - c1 - cmu + delta_hsig) * C \
                + c1 * (pc @ pc.T + (1 - hsig) * delta_hsig * C) \
                + cmu * np.sum(weights[:mu][:, np.newaxis, np.newaxis] * (y[:mu] - mean)[:, np.newaxis] @ (y[:mu] - mean)[np.newaxis, :], axis=0)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update sigma
            sigma = sigma * np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
            
            # Update mean
            mean = mean_new
            
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-6, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
