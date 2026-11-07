"""Optimized optimizer for BBOB f13 (Sharp ridge) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for f13 Sharp Ridge."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        deadline = time.time() + 280.0

        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        if lam < 4:
            lam = 4
        # Step size
        sigma = 1.0
        # Initial mean
        mean = np.zeros(dim)
        # Initial covariance matrix
        C = np.eye(dim)
        # Cholesky factor of C
        C_sqrt = np.eye(dim)
        
        # Strategy parameters
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        calls_made = 0
        
        best_x = mean.copy()
        best_fx = np.inf

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Sample offspring
            # We need to sample lam points.
            # To save time, we can generate them in a loop but vectorize the evaluation if possible? 
            # func is a black box, so we must call it individually.
            
            y_list = []
            z_list = []
            
            for i in range(lam):
                if calls_made >= budget or time.time() > deadline:
                    break
                # Sample standard normal
                z = rng.standard_normal(dim)
                # Transform to multivariate normal
                y = mean + sigma * (C_sqrt @ z)
                y_list.append(y)
                z_list.append(z)
                calls_made += 1
            
            if len(y_list) == 0:
                break
                
            # Evaluate
            f_values = []
            for y in y_list:
                fy = func(y)
                f_values.append(fy)
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

            # Sort offspring by fitness
            sorted_indices = np.argsort(f_values)
            y_sorted = [y_list[i] for i in sorted_indices]
            z_sorted = [z_list[i] for i in sorted_indices]
            f_sorted = [f_values[i] for i in sorted_indices]

            # Update mean
            # Use top mu individuals
            mean_new = np.sum(np.array([weights[i] * y_sorted[i] for i in range(mu)]), axis=0)
            
            # Update step size (CSA)
            # Standard deviation of the sampled points
            # h_sigma condition
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = 1.0 if (norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2/(dim+1)) * chi_n) else 0.0
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt.T @ np.mean(np.array([weights[i] * z_sorted[i] for i in range(mu)]), axis=0))
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance matrix
            # h_c condition
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            norm_p_c = np.linalg.norm(p_c)
            h_c = 1.0 if (norm_p_c / np.sqrt(1 - (1 - c_c)**2) < (1.4 + 2/(dim+1)) * chi_n) else 0.0
            
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean_new - mean) / sigma
            
            # Rank-one update
            v = (mean_new - mean) / sigma
            C = (1 + c_1 * h_c * (h_c - 1 + c_c) + c_mu * h_c) * C - c_1 * np.outer(p_c, p_c) + (c_1 * (h_c - 1 + c_c) + c_mu) * np.outer(v, v)
            
            # Rank-mu update
            y_diffs = np.array([y_sorted[i] for i in range(mu)])
            z_diffs = (y_diffs - mean) / sigma
            # C = C + c_mu * sum(weights[i] * outer(z_diffs[i], z_diffs[i]))
            # This is a sum of outer products
            rank_mu_term = np.sum(np.array([weights[i] * np.outer(z_diffs[i], z_diffs[i]) for i in range(mu)]), axis=0)
            C = C + c_mu * rank_mu_term
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update Cholesky factor
            # If C is not positive definite, we might need to repair it.
            # For simplicity and speed, we try Cholesky. If it fails, we regularize.
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize C
                C += 1e-6 * np.eye(dim)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C_sqrt = np.eye(dim)
                    C = np.eye(dim)
            
            mean = mean_new

        return best_x, best_fx
# EVOLVE-BLOCK-END
