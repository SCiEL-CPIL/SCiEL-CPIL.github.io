"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for continuous black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Time management
        t_start = time.time()
        deadline = t_start + 280.0  # 20s safety margin

        # CMA-ES Parameters
        # Population size
        pop_size = 4 + int(3 * np.log(dim))
        if pop_size < 4:
            pop_size = 4
        
        # Initial mean
        x_mean = np.zeros(dim)
        # Initial sigma
        sigma = 1.0
        
        # Weierstrass is often scaled, but we start with a reasonable box size
        # The box is [-5, 5]^5, so initial sigma of 1.0 covers a good portion
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Inverse square root of C (for sampling)
        # We use Cholesky decomposition for stability
        L = np.linalg.cholesky(C)
        B = np.linalg.inv(L).T
        D = np.diag(1.0 / np.sqrt(np.diag(C)))
        
        # Step size adaptation
        cs = (pop_size + 2) / (dim + pop_size + 5)
        ds = 1 + 2 * max(0, cs - 1 / (dim + 1)) + cs
        cc = (4 + dim / pop_size) / (dim + 4 + 2 * pop_size / dim)
        cmu = min(1 - cc, 2 * (pop_size - 2 + 1 / pop_size) / ((dim + 2) * (pop_size + 5)))
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Weights for selection
        # Default weights: proportional to rank
        weights = np.log(pop_size / 2) - np.log(np.arange(1, pop_size + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights[:int(pop_size/2)]**2)
        
        # Initial evaluation
        calls = 0
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls += 1
        
        def time_ok():
            return time.time() < deadline and calls < budget

        # Main loop
        while time_ok():
            # Generate population
            x_pop = np.zeros((pop_size, dim))
            f_pop = np.zeros(pop_size)
            
            for i in range(pop_size):
                if not time_ok():
                    break
                # Sample
                z = rng.standard_normal(dim)
                x_i = x_mean + sigma * (B @ (D * z))
                x_i = np.clip(x_i, -5, 5)
                x_pop[i] = x_i
                f_pop[i] = func(x_i)
                calls += 1
            
            # Sort by fitness
            idx = np.argsort(f_pop)
            x_sorted = x_pop[idx]
            f_sorted = f_pop[idx]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()
            
            # Update mean
            # Use only the top mu_eff points for mean update to avoid bias
            # Standard CMA-ES uses all points with weights, but truncation is robust
            # Here we use the weighted mean of the top half (or all, depending on weights)
            # For simplicity and robustness, let's use the top 50% weighted mean
            n_use = max(1, int(pop_size / 2))
            w_use = weights[:n_use]
            w_sum = np.sum(w_use)
            x_new_mean = np.sum(w_use[:, np.newaxis] * x_sorted[:n_use], axis=0) / w_sum
            
            # Update path p_sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (B @ (D @ ((x_new_mean - x_mean) / sigma)))
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**(2 * (calls - pop_size))) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff - 1 + 1) / (mu_eff - 1 + 1)) else 0
            # Simplified h_sigma check
            norm_p_sigma = np.linalg.norm(p_sigma)
            expected_norm = np.sqrt(mu_eff) * (1 - (1 - cs)**(calls))
            h_sigma = 1 if norm_p_sigma < (1.4 + 2/(dim+1)) * expected_norm else 0
            
            # Update path p_c
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mu_eff) * (B @ (D @ ((x_new_mean - x_mean) / sigma)))
            
            # Update covariance matrix
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(w_use[:, np.newaxis, np.newaxis] * (x_sorted[:n_use] - x_mean)[:, np.newaxis, :] @ (x_sorted[:n_use] - x_mean)[np.newaxis, :, :] / sigma**2, axis=0)
            
            C = (1 + cs * ds * (h_sigma - 1) + cc * (1 - h_sigma) - cc) * C + rank_one + cmu * rank_mu
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals = np.linalg.eigvalsh(C)
            if np.any(eigvals < 1e-12):
                C += 1e-12 * np.eye(dim)
            
            # Update sigma
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            # Clamp sigma
            sigma = np.clip(sigma, 1e-6, 100.0)
            
            # Update mean
            x_mean = x_new_mean
            
            # Update B and D
            try:
                L = np.linalg.cholesky(C)
                B = np.linalg.inv(L).T
                D = np.diag(1.0 / np.sqrt(np.diag(C)))
            except np.linalg.LinAlgError:
                # Fallback to identity if Cholesky fails
                C = np.eye(dim)
                L = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                sigma = max(sigma * 0.5, 1e-6)

        return best_x, best_f
# EVOLVE-BLOCK-END
