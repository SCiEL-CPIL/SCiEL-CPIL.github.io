import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        start_time = time.time()
        # Safety margin: 330s limit, use 325s to be safe
        deadline = start_time + 325.0

        def wrapper(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            calls_made += 1
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = func(x)
            except Exception:
                fx = float('inf')
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Random Initialization to find a good starting point
        # Use 10 calls for initial sampling
        init_calls = 10
        init_points = rng.uniform(-5, 5, (init_calls, dim))
        init_fvals = []
        for p in init_points:
            if calls_made >= total_budget:
                break
            f_val = wrapper(p)
            init_fvals.append(f_val)
        
        if calls_made >= total_budget:
            return best_x, best_f

        # Find the best initial point
        best_init_idx = np.argmin(init_fvals)
        mean = init_points[best_init_idx].copy()
        
        # Phase 2: CMA-ES
        # Parameters for CMA-ES
        lambda_pop = 10
        mu = max(1, lambda_pop // 2)
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        # Path lengths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + 4)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Damping for sigma update
        d_sigma = 1 + 2 * max(0, sqrt(mu_eff / dim - 1) - 1) + c_sigma
        
        # Pre-compute chi_n (expected norm of standard normal)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))
        
        # Cholesky decomposition update helper
        def update_chol(C, C_sqrt, c_1, c_mu, y_w, w):
            # C += c_1 * (C - I) + c_mu * sum(w_i * y_i * y_i^T)
            # We update C directly and then recompute Cholesky periodically or use a stable update.
            # For simplicity and stability in low dim, we update C and recompute Cholesky every step if dim is small.
            # But recomputing Cholesky is O(n^3). For n=5, it's negligible.
            
            # Rank-1 update for C
            # C = (1 - c_1 - c_mu) * C + c_1 * I + c_mu * sum(w_i * y_i * y_i^T)
            # Wait, the standard update is:
            # C <- (1 - c_1 - c_mu) * C + c_1 * I + c_mu * sum_{i=1}^mu w_i * y_i * y_i^T
            
            C_new = (1 - c_1 - c_mu) * C + c_1 * np.eye(dim)
            for i in range(mu):
                C_new += c_mu * weights[i] * np.outer(y_w[i], y_w[i])
            
            # Ensure symmetry
            C_new = (C_new + C_new.T) / 2.0
            
            # Regularize to avoid singularity
            eigvals, eigvecs = np.linalg.eigh(C_new)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Compute Cholesky
            try:
                C_sqrt_new = np.linalg.cholesky(C_new)
            except np.linalg.LinAlgError:
                # Fallback to identity if it fails
                C_sqrt_new = np.eye(dim)
                C_new = np.eye(dim)
                
            return C_new, C_sqrt_new

        cma_steps = 0
        max_cma_steps = (total_budget - calls_made - 50) // lambda_pop # Leave 50 for polish
        
        while cma_steps < max_cma_steps and calls_made < total_budget - 50 and time.time() < deadline:
            # Generate population
            Y = np.zeros((lambda_pop, dim))
            for i in range(lambda_pop):
                z = rng.standard_normal(dim)
                Y[i] = C_sqrt @ z
                # Transform to x-space
                x_candidate = mean + sigma * Y[i]
                # Evaluate
                f_val = wrapper(x_candidate)
                # Store y in the transformed space for covariance update
                # y_i = (x_i - mean) / sigma
                Y[i] = (x_candidate - mean) / sigma if sigma > 0 else np.zeros(dim)
                
                # We need to store the evaluated f values and the corresponding Y
                # Let's restructure to store pairs
                pass
            
            # The above loop structure is slightly inefficient for storing Y correctly if we don't keep track.
            # Let's do it properly.
            
            # Reset Y storage
            Y_list = []
            F_list = []
            X_list = []
            
            for i in range(lambda_pop):
                if calls_made >= total_budget - 50:
                    break
                z = rng.standard_normal(dim)
                y = C_sqrt @ z
                x_candidate = mean + sigma * y
                f_val = wrapper(x_candidate)
                Y_list.append(y)
                F_list.append(f_val)
                X_list.append(x_candidate)
                
            if len(F_list) == 0:
                break
            
            # Sort by fitness
            sorted_indices = np.argsort(F_list)
            Y_sorted = np.array([Y_list[i] for i in sorted_indices[:mu]])
            X_sorted = np.array([X_list[i] for i in sorted_indices[:mu]])
            
            # Update mean
            mean_new = np.sum(weights[:, np.newaxis] * X_sorted, axis=0)
            
            # Update sigma
            # y_w = (mean_new - mean) / sigma
            y_w = (mean_new - mean) / sigma
            
            # Update p_sigma
            c_sigma_sqrt = np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma_sqrt * (np.linalg.norm(y_w) / chi_n - 1) * y_w / np.linalg.norm(y_w) if np.linalg.norm(y_w) > 0 else (1 - c_sigma) * p_sigma
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (cma_steps + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update p_c
            c_c_sqrt = np.sqrt(c_c * (2 - c_c) * mu_eff)
            p_c = (1 - c_c) * p_c + h_sigma * c_c_sqrt * y_w
            
            # Update Covariance
            # y_w_matrix is Y_sorted in the transformed space?
            # No, Y_sorted are the y vectors used to generate the selected x's.
            # We need to compute the weighted sum of outer products of Y_sorted.
            
            C, C_sqrt = update_chol(C, C_sqrt, c_1, c_mu, Y_sorted, weights)
            
            mean = mean_new
            cma_steps += 1

        # Phase 3: Local Polish with (1+1)-ES
        # Use remaining budget
        remaining_budget = total_budget - calls_made
        if remaining_budget > 0 and time.time() < deadline:
            x = best_x.copy()
            fx = wrapper(x) # Ensure we evaluate the current best if not already done in the last CMA step
            # Actually, best_x is tracked in wrapper. We should re-evaluate to be safe?
            # No, wrapper updates best_x. So best_x is the current best.
            # But we need the local x to start the polish.
            x = best_x.copy()
            fx = best_f # We know this value
            
            sigma_local = 0.05
            successes = []
            window_size = 10
            steps = 0
            
            while steps < remaining_budget and calls_made < total_budget and time.time() < deadline:
                z = rng.standard_normal(dim)
                x_trial = x + sigma_local * z
                x_trial = np.clip(x_trial, -5, 5)
                
                fx_trial = wrapper(x_trial)
                
                if fx_trial < fx:
                    x = x_trial
                    fx = fx_trial
                    successes.append(1)
                else:
                    successes.append(0)
                    
                if len(successes) >= window_size:
                    success_rate = np.mean(successes[-window_size:])
                    if success_rate > 0.2:
                        sigma_local *= 1.05
                    elif success_rate < 0.2:
                        sigma_local /= 1.05
                    sigma_local = np.clip(sigma_local, 1e-6, 1.0)
                    
                steps += 1

        return best_x, best_f
