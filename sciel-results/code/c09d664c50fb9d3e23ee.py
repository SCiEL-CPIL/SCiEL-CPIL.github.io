import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill
        
        # We will use a robust CMA-ES implementation from scratch.
        # Since the budget is small (500) and dim is small (5), we can afford a full CMA-ES.
        # We will use a multi-restart strategy: run CMA-ES with a portion of the budget,
        # and if it stagnates or finishes, restart with a new random center if time/budget allows.
        
        # However, given the specific structure of Lunacek (bimodal), a single run of CMA-ES
        # with a good initial sigma and mean is often sufficient if the initial point is close.
        # But "close" is hard to know. 
        # A better strategy for small budgets is a "1+1" CMA-ES (which is just (1+1)-ES with covariance adaptation) 
        # or a small population CMA-ES.
        
        # Let's implement a standard CMA-ES with lambda=4 (small for speed and exploration in 500 calls).
        # Actually, for 500 calls, lambda=4 means 125 generations.
        
        def cma_es_run(func, budget_remaining, rng, start_x, start_sigma):
            # Returns (best_x, best_f, calls_used)
            # State variables
            x_mean = start_x.copy()
            sigma = start_sigma
            C = np.eye(self.dim)
            pc = np.zeros(self.dim)
            ps = np.zeros(self.dim)
            
            # Hyperparameters
            lambda_ = 4
            mu = lambda_ // 2
            
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Step size adaptation
            c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
            
            # Covariance matrix adaptation
            c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
            c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
            
            chi_n = np.sqrt(self.dim) * (1 - 1.0 / (4 * self.dim) + 1.0 / (21 * self.dim**2))
            
            best_x = x_mean.copy()
            best_f = np.inf
            calls = 0
            
            # Precompute L (Cholesky) initially
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(self.dim)
                L = np.eye(self.dim)
            
            while calls < budget_remaining:
                # Check time
                if time.time() > deadline:
                    break
                
                # Generate population
                Z = rng.standard_normal((lambda_, self.dim))
                Y = x_mean[None, :] + sigma * (Z @ L)
                
                # Clip to box
                Y = np.clip(Y, -5, 5)
                
                # Evaluate
                # We need to ensure we don't exceed budget
                if calls + lambda_ > budget_remaining:
                    # Evaluate only as many as we can
                    num_eval = budget_remaining - calls
                    Y_eval = Y[:num_eval]
                else:
                    Y_eval = Y
                    num_eval = lambda_
                
                f_vals = np.array([func(x) for x in Y_eval])
                calls += num_eval
                
                # Update best
                idx_min = np.argmin(f_vals)
                if f_vals[idx_min] < best_f:
                    best_f = f_vals[idx_min]
                    best_x = Y_eval[idx_min].copy()
                
                # If we didn't evaluate a full population, we can't update CMA-ES parameters properly.
                # For the last few steps, we just stop.
                if num_eval < lambda_:
                    break
                
                # Sort
                idx = np.argsort(f_vals)
                Y_sorted = Y_eval[idx]
                f_sorted = f_vals[idx]
                
                # Update mean
                x_mean_old = x_mean.copy()
                x_mean_new = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)
                
                # Update path covariances
                # pc = (1 - c_c) * pc + c_c * (x_mean_new - x_mean_old) / sigma
                pc = (1 - c_c) * pc + c_c * (x_mean_new - x_mean_old) / sigma
                
                # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (x_mean_new - x_mean_old)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (x_mean_new - x_mean_old)
                
                # Update sigma
                h_sigma = (1 - np.exp(-1 / (c_sigma * (lambda_ - 1)))) if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1))) < (1.4 + 2 / (self.dim + 1)) * chi_n else 0
                sigma *= np.exp(c_sigma * d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update covariance
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C \
                    + c_1 * (pc[:, None] * pc[None, :] + (1 - h_sigma) * c_c * (2 - c_c) * np.eye(self.dim) - np.eye(self.dim)) \
                    + c_mu * np.sum(weights[:, None, None] * np.outer(Y_sorted[:mu] - x_mean_old, Y_sorted[:mu] - x_mean_old) / (sigma**2), axis=0)
                
                # Regularize C to ensure positive definiteness
                # Simple check: if eigenvalues are negative, reset
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Reset C to identity or previous valid state
                    # A robust fallback: reset to identity
                    C = np.eye(self.dim)
                    L = np.eye(self.dim)
                    pc = np.zeros(self.dim)
                    ps = np.zeros(self.dim)
                    sigma = start_sigma * 0.5 # Reduce step size
            
                x_mean = x_mean_new
            
            return best_x, best_f, calls

        # Multi-restart strategy
        # We will split the budget into chunks.
        # Given 500 calls, let's try 3 restarts of ~160 calls each, or 2 restarts of 250.
        # Lunacek has two global minima. One is at the origin (shifted), the other is at a specific offset.
        # A restart helps if the first run gets stuck in the wrong basin.
        
        # However, CMA-ES is quite good at escaping local minima if sigma is adapted well.
        # Let's try a single run first, but with a large initial sigma to cover the box.
        # If we have time/budget left, we can do a local polish.
        
        # Actually, the most robust strategy for small budgets on BBOB is often a single CMA-ES run
        # with appropriate initialization.
        
        rng = np.random.default_rng()
        
        # Initial mean: center of the box
        x_start = np.zeros(self.dim)
        # Initial sigma: large enough to explore the box
        sigma_start = 2.5
        
        # Run CMA-ES
        best_x, best_f, calls_used = cma_es_run(func, self.budget, rng, x_start, sigma_start)
        
        self.best_x = best_x
        self.best_f = best_f
        self.calls = calls_used
        
        # If we have budget left and time, do a local polish with Nelder-Mead or similar?
        # Nelder-Mead is not great for non-smooth or noisy, but Lunacek is smooth (Rastrigin part is smooth).
        # However, Rastrigin has many local minima. Nelder-Mead might get stuck.
        # CMA-ES is already doing local refinement.
        
        # If we have significant budget left (e.g., > 50 calls), we could try a second CMA-ES run
        # centered at the best found so far, with a smaller sigma.
        remaining = self.budget - self.calls
        if remaining > 50 and time.time() < deadline - 10:
            # Restart around best found
            x_start2 = self.best_x
            sigma_start2 = 0.5 # Smaller step
            best_x2, best_f2, calls_used2 = cma_es_run(func, remaining, rng, x_start2, sigma_start2)
            
            if best_f2 < self.best_f:
                self.best_x = best_x2
                self.best_f = best_f2
            self.calls += calls_used2
            
        return self.best_x, self.best_f
