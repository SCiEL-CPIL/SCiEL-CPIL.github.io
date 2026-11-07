import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        
        # Use a fixed seed for internal randomness to ensure reproducibility 
        # if the evaluator re-runs, but typically random is fine. 
        # However, since we don't know the instance index, a fixed seed 
        # might be risky if the landscape is symmetric. 
        # Let's use a random seed based on time or just default.
        # The prompt implies the evaluator creates the instance. 
        # I will use a random seed to avoid systematic bias.
        rng = np.random.default_rng()
        
        best_x = None
        best_f = float('inf')
        
        # CMA-ES Parameters
        # Population size lambda
        lambda_ = 4 + int(3 * np.log(dim))
        if lambda_ < 2:
            lambda_ = 4
        # Number of recombination points
        mu = lambda_ // 2
        if mu < 1:
            mu = 1
            
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State variables
        # Initialize mean randomly in the box
        x_mean = rng.uniform(-5, 5, dim)
        sigma = 2.0
        C = np.eye(dim)
        B = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        evals_left = budget
        generation = 0
        
        while evals_left > 0:
            if time.time() > deadline:
                break
                
            # Check if we have enough budget for a full generation
            if evals_left < lambda_:
                # Evaluate remaining points one by one
                # Generate remaining population
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    B = np.eye(dim)
                    
                z = rng.standard_normal((evals_left, dim))
                y = sigma * (z @ B.T)
                x_pop = x_mean[None, :] + y
                
                for i in range(evals_left):
                    fx = func(x_pop[i])
                    if fx < best_f:
                        best_f = fx
                        best_x = x_pop[i].copy()
                    evals_left -= 1
                break
            
            # Generate full population
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Reset if covariance matrix becomes invalid
                C = np.eye(dim)
                B = np.eye(dim)
                
            z = rng.standard_normal((lambda_, dim))
            y = sigma * (z @ B.T)
            x_pop = x_mean[None, :] + y
            
            # Evaluate population
            f_pop = np.array([func(x_pop[i]) for i in range(lambda_)])
            evals_left -= lambda_
            generation += 1
            
            # Sort by fitness
            idx = np.argsort(f_pop)
            x_sorted = x_pop[idx]
            f_sorted = f_pop[idx]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()
            
            # Recombination
            x_new = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            
            # Update step size (sigma)
            # Calculate C^{-1/2} (x_new - x_mean) / sigma
            # Using eigen decomposition for small dim
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            d_sigma_vec = C_inv_sqrt @ ((x_new - x_mean) / sigma)
            
            # h_sigma heuristic
            # ||p_sigma|| / sqrt(1 - (1 - c_sigma)^{2 gen})
            # Approximate threshold
            h_sigma = 1.0
            # Simple check: if p_sigma norm is too large relative to expected
            # E||N(0, I)|| = sqrt(dim) * (1 - 1/(4dim) + ...) approx sqrt(dim)
            # Threshold is (1 + 2/(dim+1)) * sqrt(dim)
            threshold = (1 + 2.0 / (dim + 1)) * np.sqrt(dim)
            # We need the norm of p_sigma in the normalized space
            # p_sigma is already in the normalized space (it accumulates normalized steps)
            if np.linalg.norm(p_sigma) > threshold:
                h_sigma = 0.0
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * d_sigma_vec
            
            # Update p_c
            # p_c = (1 - c_c) p_c + h_sigma * sqrt(c_c (2 - c_c) mu_eff) * (x_new - x_mean) / sigma
            # Note: The vector here is the raw step divided by sigma, not C^{-1/2}
            d_c_vec = (x_new - x_mean) / sigma
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * d_c_vec
            
            # Update Covariance Matrix C
            # C = (1 + c_1 (2 - c_1) - c_mu) C 
            #      + c_1 (p_c p_c^T - (1 - c_c) C)
            #      + c_mu sum(weights_i (y_i y_i^T))
            
            # y_i = (x_sorted[i] - x_old) / sigma
            # x_old is x_mean before update
            x_old = x_mean
            y_sorted = (x_sorted[:mu] - x_old[None, :]) / sigma
            
            # Rank-1 update
            rank1 = p_c[:, None] * p_c[None, :] - (1 - c_c) * C
            
            # Rank-mu update
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * (y_sorted[i, :, None] * y_sorted[i, None, :])
            
            C = (1 + c_1 * (2 - c_1) - c_mu) * C + c_1 * rank1 + c_mu * rank_mu
            
            # Update mean
            x_mean = x_new
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(dim) - 1))
            
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # If sigma is very small, we might be converged
            if sigma < 1e-4:
                break

        # Polishing phase with remaining budget
        remaining = evals_left
        if remaining > 0 and best_x is not None and time.time() < deadline:
            # Simple (1+1)-ES polish
            x = best_x.copy()
            fx = best_f
            sigma_polish = max(sigma, 0.1) # Start with current sigma or 0.1
            # Actually, use a smaller sigma for polishing
            sigma_polish = 0.5
            
            successes = 0
            attempts = 0
            window_size = 5
            
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                
                y = x + sigma_polish * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = x.copy()
                
                attempts += 1
                
                if attempts % window_size == 0:
                    success_rate = successes / attempts
                    if success_rate > 0.2:
                        sigma_polish *= 1.2
                    elif success_rate < 0.2:
                        sigma_polish *= 0.8
                    sigma_polish = np.clip(sigma_polish, 0.001, 1.0)
                    successes = 0
                    attempts = 0

        return best_x, best_f
