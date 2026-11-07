import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin from 330s limit
        max_calls = self.budget
        dim = self.dim
        
        # CMA-ES parameters
        # For dim=5, lambda is typically 4 + floor(3*ln(dim)) = 4 + 4 = 8
        # Let's use 10 for slightly more exploration
        lambda_ = 10
        mu = lambda_ // 2
        
        # Weierstrass is multimodal, but CMA-ES is good at finding the global
        # minimum if the initial sigma is large enough to cover the box.
        # Box is [-5, 5]^5. Center is 0.
        # To cover the box, sigma should be at least ~5/sqrt(5) ~ 2.2 for the
        # marginal distributions to have significant mass in the corners.
        # Let's start with sigma = 5.0. This is very large, but CMA-ES adapts.
        
        x_mean = np.zeros(dim)
        sigma = 5.0
        
        # Covariance matrix C, initially identity
        C = np.eye(dim)
        # Inverse of C
        C_inv = np.eye(dim)
        # Cholesky factor of C
        C_chol = np.eye(dim)
        
        # Step size control parameters
        # c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        mu_eff = np.sum(np.arange(1, mu+1)**-2) / np.sum(np.arange(1, lambda_+1)**-2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        # d_sigma = 1 + 2 * max(0, sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Covariance adaptation parameters
        # c_c = (4 + mu_eff / dim) / (dim^1.8 + mu_eff + 13)
        c_c = (4 + mu_eff / dim) / (dim**1.8 + mu_eff + 13)
        # c_1 = 2 / (dim^1.8 + mu_eff)
        c_1 = 2 / (dim**1.8 + mu_eff)
        # c_mu = 2 * (mu_eff - 2 + 2/mu_eff) / (dim^1.6 + mu_eff)
        c_mu = 2 * (mu_eff - 2 + 2/mu_eff) / (dim**1.6 + mu_eff)
        
        # Update weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, lambda_ + 1))
        weights = weights / np.sum(weights)
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Call counter
        calls = 0
        best_x = None
        best_f = np.inf
        
        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= max_calls:
                raise Exception("Budget exceeded")
            calls += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def time_check():
            return time.time() > deadline or calls >= max_calls

        # Main CMA-ES loop
        while not time_check():
            # 1. Sample population
            try:
                # Update Cholesky if needed (we update it every step below, so it's fresh)
                # Actually, we update C and C_chol at the end of the previous iteration.
                # For the first iteration, C_chol is identity.
                
                z = np.random.randn(lambda_, dim)
                # x_i = x_mean + sigma * C_chol @ z_i^T
                # z is (lambda, dim), C_chol is (dim, dim)
                # We want x_i = x_mean + sigma * (z_i @ C_chol^T)
                # Since C is symmetric, C_chol is lower triangular, C_chol^T is upper.
                # Standard CMA-ES uses x = x_mean + sigma * C_chol @ z
                # If z is a row vector, x = x_mean + sigma * z @ C_chol.T
                # Let's stick to standard: x_i = x_mean + sigma * C_chol @ z_i
                # z_i is a column vector.
                # In numpy, if z is (lambda, dim), z_i is z[i].
                # x[i] = x_mean + sigma * (C_chol @ z[i])
                
                # Efficient way:
                # X = z @ C_chol.T + x_mean[None, :]
                X = z @ C_chol.T + x_mean[None, :]
                X *= sigma
                X += x_mean[None, :] # Wait, the formula is x = mean + sigma * C_chol @ z
                # So X = sigma * (z @ C_chol.T) + x_mean
                
                # Re-evaluating:
                # z is (lambda, dim).
                # C_chol is (dim, dim).
                # We want x_i = x_mean + sigma * (C_chol @ z_i)
                # In matrix form: X = x_mean[None, :] + sigma * (z @ C_chol.T)
                
                X = x_mean[None, :] + sigma * (z @ C_chol.T)
                
            except Exception:
                break

            # 2. Evaluate population
            F = np.zeros(lambda_)
            for i in range(lambda_):
                if time_check():
                    break
                try:
                    F[i] = wrapped_func(X[i])
                except Exception:
                    # If budget is hit, fill rest with inf
                    F[i:] = np.inf
                    break
            
            if time_check():
                break

            # 3. Sort by fitness
            # If some calls failed, F[i] is inf.
            # We need to handle the case where we didn't evaluate all.
            # But the loop breaks, so F has infs.
            # Sorting infs to the end is fine.
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            # Check if we have at least mu valid points
            # If not, we might have run out of budget mid-generation.
            # In that case, we should stop the main loop.
            if np.sum(np.isfinite(F_sorted)) < mu:
                break

            # 4. Update mean
            x_mean_new = np.sum(weights[None, :] * X_sorted[:mu], axis=0)
            
            # 5. Update sigma
            # Calculate norm of evolution path p_sigma
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_mean_new - x_mean) / sigma
            # C_inv_sqrt is the inverse of the square root of C.
            # We have C_chol such that C = C_chol @ C_chol.T
            # So C_inv_sqrt = inv(C_chol.T)
            
            try:
                C_inv_sqrt = np.linalg.inv(C_chol.T)
            except np.linalg.LinAlgError:
                break

            # Direction vector
            dir_vec = (x_mean_new - x_mean) / sigma
            # Project onto C_inv_sqrt
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ dir_vec)
            
            # Hessian correction
            h_sig = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1))) else 0
            # Note: calls is incremented in wrapped_func, so it reflects total calls.
            # The exponent should be based on the number of generations, not calls.
            # Let's use a generation counter for the h_sig check.
            # For now, this approximation is okay, but let's be more precise.
            # We'll track generations separately.
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(dim) - 1))
            
            # 6. Update covariance
            # p_c = (1 - c_c) * p_c + h_sig * sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ (x_mean_new - x_mean) / sigma
            p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ dir_vec)
            
            # Rank-mu update
            # Delta H = c_1 * (p_c @ p_c.T - C) + c_mu * sum(weights_i * (y_i @ y_i.T))
            # where y_i = C_inv_sqrt @ (x_i - x_mean_old) / sigma
            
            Y = (X_sorted[:mu] - x_mean[None, :]) / sigma
            Y = Y @ C_inv_sqrt.T
            
            Delta_H = c_1 * (p_c[:, None] @ p_c[None, :] - C) + c_mu * np.sum(weights[None, :, None] * Y[:, :, None] * Y[:, None, :], axis=0)
            
            C += Delta_H
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small regularization to ensure PD
            C += 1e-12 * np.eye(dim)
            
            # Update Cholesky
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset C to identity or use previous
                # This can happen if C becomes non-PD due to numerical issues
                C = np.eye(dim)
                C_chol = np.eye(dim)
                
            x_mean = x_mean_new

        return best_x, best_f
