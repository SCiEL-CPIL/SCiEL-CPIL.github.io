import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def can_call():
            nonlocal calls_made
            if calls_made >= budget:
                return False
            if time.time() > deadline:
                return False
            return True

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if not can_call():
                return float('inf')
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Strategy: Two-Phase Approach
        # Phase 1: Wide Exploration with a large population and large initial sigma.
        # This is designed to quickly capture the anisotropic structure of the Discus function.
        # Phase 2: High-Precision Refinement.
        # Take the best point found in Phase 1 and run a standard CMA-ES with a smaller population
        # and adaptive sigma to polish the solution.

        # --- Phase 1: Exploration ---
        pop_size_1 = 20
        max_generations_1 = 15  # 15 * 20 = 300 calls
        sigma_1 = 3.0           # Large sigma to cover the domain [-5, 5]
        
        # Initialize CMA-ES state for Phase 1
        x_mean = np.zeros(dim)
        sigma = sigma_1
        C = np.eye(dim)
        p_c = np.zeros(dim)
        p_s = np.zeros(dim)
        
        # Parameters for Phase 1
        mu = pop_size_1 // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_c = min(c_c, 1.0)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (weights[0] - 1 + weights[1]) / ((dim + 2)**2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))
        
        # Run Phase 1
        for gen in range(max_generations_1):
            if not can_call():
                break
                
            # Generate population
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            B = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            z = np.random.randn(pop_size_1, dim)
            samples = x_mean[None, :] + sigma * (z @ B.T)
            
            f_values = np.zeros(pop_size_1)
            valid_count = 0
            for i in range(pop_size_1):
                if not can_call():
                    break
                f_values[i] = evaluate(samples[i])
                valid_count += 1
            
            if valid_count < mu:
                break
                
            # Sort
            order = np.argsort(f_values[:valid_count])
            x_selected = samples[:valid_count][order]
            f_selected = f_values[:valid_count][order]
            
            # Update mean
            x_mean_new = np.sum(weights[:mu, None] * x_selected[:mu], axis=0)
            
            # Update step size control variable
            B_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            y_mean = (x_mean_new - x_mean) / sigma
            p_s = (1 - c_sigma) * p_s + c_sigma * np.sqrt(mu_eff) * (B_inv @ y_mean)
            
            norm_ps = np.linalg.norm(p_s)
            h_sigma = 1.0 if norm_ps < 1.75 * chi_n else 0.0
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * c_c * y_mean
            
            # Compute y_i
            y_i = (x_selected[:mu] - x_mean) / sigma
            sum_outer = np.zeros((dim, dim))
            for i in range(mu):
                sum_outer += weights[i] * np.outer(y_i[i], y_i[i])
                
            # Update C
            # Approximate delta term for stability
            delta_term = 1.0
            c_C = 1 - c_1 - c_mu + c_mu * h_sigma * delta_term
            C = c_C * C + c_1 * np.outer(p_c, p_c) + c_mu * sum_outer
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (norm_ps - chi_n))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update mean
            x_mean = x_mean_new
            
            if sigma < 1e-6:
                break

        # --- Phase 2: Refinement ---
        # Use the best point found so far
        x_start = best_x.copy()
        
        # Parameters for Phase 2
        pop_size_2 = 8
        sigma_2 = 0.5  # Smaller sigma for refinement
        
        x_mean = x_start
        sigma = sigma_2
        C = np.eye(dim)
        p_c = np.zeros(dim)
        p_s = np.zeros(dim)
        
        mu = pop_size_2 // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_c = min(c_c, 1.0)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (weights[0] - 1 + weights[1]) / ((dim + 2)**2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))
        
        # Run Phase 2 until budget or time is exhausted
        while can_call():
            # Generate population
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            B = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            z = np.random.randn(pop_size_2, dim)
            samples = x_mean[None, :] + sigma * (z @ B.T)
            
            f_values = np.zeros(pop_size_2)
            valid_count = 0
            for i in range(pop_size_2):
                if not can_call():
                    break
                f_values[i] = evaluate(samples[i])
                valid_count += 1
            
            if valid_count < mu:
                break
                
            # Sort
            order = np.argsort(f_values[:valid_count])
            x_selected = samples[:valid_count][order]
            f_selected = f_values[:valid_count][order]
            
            # Update mean
            x_mean_new = np.sum(weights[:mu, None] * x_selected[:mu], axis=0)
            
            # Update step size control variable
            B_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            y_mean = (x_mean_new - x_mean) / sigma
            p_s = (1 - c_sigma) * p_s + c_sigma * np.sqrt(mu_eff) * (B_inv @ y_mean)
            
            norm_ps = np.linalg.norm(p_s)
            h_sigma = 1.0 if norm_ps < 1.75 * chi_n else 0.0
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * c_c * y_mean
            
            # Compute y_i
            y_i = (x_selected[:mu] - x_mean) / sigma
            sum_outer = np.zeros((dim, dim))
            for i in range(mu):
                sum_outer += weights[i] * np.outer(y_i[i], y_i[i])
                
            # Update C
            delta_term = 1.0
            c_C = 1 - c_1 - c_mu + c_mu * h_sigma * delta_term
            C = c_C * C + c_1 * np.outer(p_c, p_c) + c_mu * sum_outer
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (norm_ps - chi_n))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update mean
            x_mean = x_mean_new
            
            if sigma < 1e-6:
                break

        return best_x
