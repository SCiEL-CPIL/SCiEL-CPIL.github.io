"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Black-box continuous optimizer for BBOB f22 (Gallagher 21 peaks).
    Uses Restart CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    
    CMA-ES adapts the multivariate normal distribution used to sample candidates
    by updating a covariance matrix. This allows it to handle correlations and
    ill-conditioned landscapes effectively. Restart mechanism helps escape
    local minima in multimodal functions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        n_evals = 0

        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (lambda_ / 2 + 1) / (dim + lambda_ / 2 + 5)
        c_c = 4 / (dim + 4)
        c_1 = 2 / ((dim + 1.3)**2 + lambda_)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + lambda_))
        
        sigma0 = 1.0
        max_generations_per_restart = 60
        stagnation_threshold = 25
        min_sigma = 1e-6
        max_sigma = 10.0

        while n_evals < budget:
            # Initialize a new run
            if best_x is None:
                x_mean = rng.uniform(-5, 5, size=dim)
            else:
                # Random restart for diversity
                x_mean = rng.uniform(-5, 5, size=dim)
            
            C = np.eye(dim)
            sigma = sigma0
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 + 1/(4*dim) - 1/(21*dim**2))
            
            best_f_run = np.inf
            best_f_run_prev = np.inf
            stagnation_count = 0
            gen = 0
            
            while n_evals < budget and gen < max_generations_per_restart:
                # Compute Cholesky factor of C for sampling
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset C to identity or add small regularization
                    C = np.eye(dim) + 1e-6 * np.eye(dim)
                    L = np.linalg.cholesky(C)
                
                # Generate population
                x_offspring = np.zeros((lambda_, dim))
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    x_offspring[i] = x_mean + sigma * (L @ z)
                
                # Evaluate population
                f_values = np.zeros(lambda_)
                for i in range(lambda_):
                    if n_evals >= budget:
                        break
                    x_candidate = np.clip(x_offspring[i], -5, 5)
                    f_candidate = func(x_candidate)
                    n_evals += 1
                    f_values[i] = f_candidate
                    
                    if f_candidate < best_f:
                        best_f = f_candidate
                        best_x = x_candidate.copy()
                    if f_candidate < best_f_run:
                        best_f_run = f_candidate
                
                if n_evals >= budget:
                    break
                
                # Sort indices by fitness
                indices = np.argsort(f_values)
                x_selected = x_offspring[indices[:mu]]
                
                # Stagnation check
                if best_f_run < best_f_run_prev - 1e-8:
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                best_f_run_prev = best_f_run
                
                # Update mean
                x_new_mean = np.sum(weights[:, None] * x_selected, axis=0)
                x_old_mean = x_mean
                x_mean = x_new_mean
                
                # Update evolution paths
                # Calculate h_sigma
                norm_ps = np.linalg.norm(ps)
                denom = np.sqrt(1 - (1 - c_sigma)**(2*(gen+1)))
                if denom > 1e-10:
                    hsig = (norm_ps / denom) / chi_n < 1.4 + 2/(dim+1)
                else:
                    hsig = 0
                
                # Update ps
                L_inv = np.linalg.inv(L)
                diff = x_new_mean - x_old_mean
                ps = (1 - c_sigma) * ps + hsig * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L_inv @ diff) / sigma
                
                # Update pc
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff / sigma
                
                # Update covariance matrix
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                
                # Rank one update
                rank_one = np.outer(diff, diff) / sigma**2
                
                # Rank mu update
                diff_selected = x_selected - x_old_mean
                scaled_diff = diff_selected / sigma
                rank_mu = (scaled_diff.T * weights) @ scaled_diff
                
                C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + delta_hsig) + c_mu * rank_mu
                
                # Symmetrize
                C = (C + C.T) / 2
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, min_sigma, max_sigma)
                
                gen += 1
                
                # Restart conditions
                if sigma < min_sigma:
                    break
                if stagnation_count > stagnation_threshold:
                    break

            if n_evals >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
