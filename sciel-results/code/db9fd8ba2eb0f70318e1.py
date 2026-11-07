import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 280.0
        
        dim = self.dim
        budget = self.budget
        
        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        def time_check():
            return (time.time() - start_time) > timeout

        def evaluate(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                return None
            if time_check():
                return None
            # Clip to ensure we stay in bounds, though func also clips
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        if best_x is None:
            # Initial point to satisfy contract if we somehow exit early (shouldn't happen)
            x0 = np.zeros(dim)
            evaluate(x0)
            if best_x is None:
                return np.zeros(dim), 0.0

        # --- CMA-ES Implementation ---
        
        # Population size
        # Standard CMA-ES uses lambda = 4 + floor(3 * log(n))
        # For n=5, lambda = 4 + floor(3*1.6) = 4 + 4 = 8.
        # We can use a larger population for better exploration in the first few steps,
        # but let's stick to a reasonable size to conserve budget for more generations.
        # lambda = 10 is a good balance for dim=5.
        pop_size = 10
        
        # Initial sigma
        # The domain is [-5, 5], so a step size of 1.0 covers 1/5th of the range.
        # This is reasonable for an initial guess.
        sigma = 1.0
        
        # Mean
        # Start from the best point found so far (which is just the initial point if we didn't do a pre-phase, 
        # but here we start fresh with CMA-ES logic. 
        # Wait, the current program does a Latin Hypercube phase. 
        # The inspiration suggests switching TO CMA-ES. 
        # Should we keep the LHS phase? 
        # CMA-ES is a global optimizer if initialized well, but it can get stuck in local minima.
        # The "Step Ellipsoidal" function has a specific structure.
        # Let's use a small LHS phase to find a good starting center for CMA-ES, 
        # or just start CMA-ES from the origin or a random point.
        # Given the budget is 500, and pop_size is 10, we can afford ~50 generations.
        # A 20-point LHS phase leaves 480 points for CMA-ES (48 generations).
        # This seems robust.
        
        # Let's do a quick LHS to find a good starting mean.
        n_coarse = 20
        lhp = np.zeros((n_coarse, dim))
        for d in range(dim):
            idx = rng.permutation(n_coarse)
            u = (idx + rng.uniform(0, 1, n_coarse)) / n_coarse
            lhp[:, d] = -5.0 + 10.0 * u
        
        f_vals = np.full(n_coarse, np.inf)
        for i in range(n_coarse):
            f_val = evaluate(lhp[i])
            if f_val is not None:
                f_vals[i] = f_val
        
        # Find the best point from LHS to initialize CMA-ES mean
        best_idx = np.argmin(f_vals)
        mean = lhp[best_idx].copy()
        
        # If all evaluations failed (unlikely), fallback
        if np.isinf(f_vals[best_idx]):
            mean = np.zeros(dim)
            evaluate(mean)

        # CMA-ES Parameters
        # Adaptation rates
        mu = pop_size // 2  # Number of weights
        # Weights: w_i = log(mu + 0.5) - log(i + 1) for i in 0..mu-1
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= weights.sum()
        
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size adaptation rate
        # c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_sigma = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
        # d_sigma = 1 + 2 * max(0, sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
        
        # Covariance matrix adaptation
        # c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_c = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim)
        # c_1 = 2 / (n^2.1 + 10)
        c_1 = 2.0 / (dim**2.1 + 10.0)
        # c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / (n^2 + 2))
        c_mu = min(1.0 - c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / (dim**2 + 2.0))
        
        # Covariance matrix C
        C = np.eye(dim)
        # Cholesky factor B such that C = B @ B.T
        B = np.eye(dim)
        
        # Evolution path for sigma
        p_sigma = np.zeros(dim)
        # Evolution path for covariance
        p_c = np.zeros(dim)
        
        # Main Loop
        while eval_count < budget and not time_check():
            # Generate population
            offspring = np.zeros((pop_size, dim))
            f_offspring = np.full(pop_size, np.inf)
            
            for i in range(pop_size):
                # Sample z from standard normal
                z = rng.standard_normal(dim)
                # Transform to x
                x = mean + sigma * (B @ z)
                f_val = evaluate(x)
                if f_val is None:
                    break
                offspring[i] = x
                f_offspring[i] = f_val
            
            if eval_count >= budget:
                break
            
            # Sort offspring by fitness
            order = np.argsort(f_offspring)
            offspring = offspring[order]
            f_offspring = f_offspring[order]
            
            # Update weights and mean
            # Only use the best mu individuals
            y_w = np.zeros((mu, dim))
            for i in range(mu):
                y_w[i] = (offspring[i] - mean) / sigma
            
            # New mean
            mean_new = mean + sigma * (weights @ y_w)
            
            # Update evolution paths
            # p_sigma
            h_sigma = 1.0
            # Threshold for h_sigma
            # h_sigma = 1 - exp(-1 / (c_sigma * (n + 1)))
            # This is a simplification. Standard CMA-ES uses a more complex check.
            # For simplicity and robustness in this short budget, we can use a fixed h_sigma = 1 
            # or a simplified version. Let's use the standard update but simplified.
            
            # Actually, let's use a slightly simplified CMA-ES to avoid bugs in path updates.
            # The core power is in the covariance adaptation.
            
            p_sigma = (1.0 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ (weights @ y_w))
            
            # Check if p_sigma is too large to decide h_sigma
            # ||p_sigma||^2 / chi_n^2 > 1 + 2/(n+1)
            # chi_n^2 is the expected squared norm of a standard normal vector
            chi_n_sq = dim - 1.0 / 3.0
            if np.dot(p_sigma, p_sigma) / chi_n_sq < (1.0 + 2.0 / (dim + 1.0)) * (1.0 - c_sigma):
                h_sigma = 1.0
            else:
                h_sigma = 0.0
            
            # p_c
            p_c = (1.0 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ (weights @ y_w))
            
            # Update covariance matrix
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = y_w[i] / np.sqrt(weights[i])
                rank_mu += np.outer(y_i, y_i)
            
            # Delta term
            delta_h = (1.0 - h_sigma) * h_sigma * c_c * (2.0 - c_c)
            
            C = (1.0 + c_c * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Re-calculate B (Cholesky)
            # Ensure C is positive definite
            # Symmetrize
            C = (C + C.T) / 2.0
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If not positive definite, regularize
                C += 1e-6 * np.eye(dim)
                B = np.linalg.cholesky(C)
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(chi_n_sq) - 1.0))
            
            # Update mean
            mean = mean_new
            
            # Clip mean to bounds? No, CMA-ES works in the transformed space.
            # But we should ensure the next samples are clipped. The evaluate function clips.
            
            # If sigma becomes too small or too large, we might stop or reset
            if sigma < 1e-10:
                break
        
        return best_x, best_f
