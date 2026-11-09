"""Optimizer for BBOB f6 (Attractive sector) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Continuous optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls_made = 0
        x_best = None
        f_best = np.inf

        def eval_func(x):
            nonlocal calls_made, x_best, f_best
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < f_best:
                f_best = f
                x_best = x.copy()
            return f

        # CMA-ES Parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        # Weights for rank-mu update
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Step size adaptation
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0 # Initial step size
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute square root of C for sampling
        # We will update this every step
        
        while calls_made < budget:
            # Generate population
            pops = []
            f_vals = []
            
            # Determine how many we can evaluate in this generation
            # We can evaluate at most budget - calls_made
            # But we also need to respect the lambda_ size
            # If we can't fit a full generation, we might do a partial one or stop
            # For simplicity, if remaining budget < lambda_, we just evaluate remaining points one by one?
            # No, CMA-ES works best with full generations.
            # If remaining < lambda_, we stop the main loop and just do local search?
            # Or just evaluate the rest?
            # Let's just run full generations as long as we have at least lambda_ calls left.
            
            if budget - calls_made < lambda_:
                break

            for i in range(lambda_):
                # Sample from N(0, C)
                # Use Cholesky decomposition or eigen decomposition
                # For efficiency, we can use the fact that C is symmetric positive definite
                # But for small dim, Cholesky is fine.
                try:
                    L = np.linalg.cholesky(C)
                    z = rng.standard_normal(dim)
                    y = L @ z
                except np.linalg.LinAlgError:
                    # Fallback if C is not positive definite (shouldn't happen often)
                    y = rng.standard_normal(dim)
                
                x_i = x_mean + sigma * y
                x_i = np.clip(x_i, -5, 5)
                
                f_i = eval_func(x_i)
                pops.append(x_i)
                f_vals.append(f_i)
                
                if calls_made >= budget:
                    break

            if calls_made >= budget:
                break

            # Sort population by fitness
            sort_idx = np.argsort(f_vals)
            pops = [pops[i] for i in sort_idx]
            f_vals = [f_vals[i] for i in sort_idx]
            
            # Update mean
            # x_mean_new = sum(weights[i] * pops[i] for i in range(mu))
            x_mean_new = np.zeros(dim)
            for i in range(mu):
                x_mean_new += weights[i] * pops[i]
            
            # Update path states
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (x_mean_new - x_mean)
            # Note: The formula usually uses the standardized difference
            # Let's use the standard update rules
            
            # Calculate the difference in the standardized space
            # d = (x_mean_new - x_mean) / sigma
            d = (x_mean_new - x_mean) / sigma if sigma > 0 else np.zeros(dim)
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * d
            
            # Heuristic to prevent divergence
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made // lambda_ + 1))) > (1.4 + 2 / (dim + 1)) * chi_n:
                c_sigma_eff = 0.0
            else:
                c_sigma_eff = c_sigma
            
            # Update pc
            # pc = (1 - c_c) * pc + h_sig * sqrt(c_c * (2 - c_c) * mu_eff) * d
            # h_sig = 1 if ||ps|| / sqrt(1 - (1-c_sigma)^(2*gen)) < (1.4 + 2/(dim+1)) * chi_n
            # We approximated gen above. Let's use a simpler heuristic or just assume h_sig=1 for now?
            # Standard CMA-ES uses h_sig.
            gen = calls_made // lambda_
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n:
                h_sig = 1.0
            else:
                h_sig = 0.0
                
            pc = (1 - c_c) * pc + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * d
            
            # Update covariance matrix
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (1 - h_sig) * c_c * (2 - c_c) * C) + c_mu * sum(weights[i] * y_i * y_i.T)
            
            # Calculate y_i for the top mu individuals
            # y_i = (x_i - x_mean) / sigma
            sum_yy = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (pops[i] - x_mean) / sigma if sigma > 0 else np.zeros(dim)
                sum_yy += weights[i] * np.outer(y_i, y_i)
                
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - h_sig) * c_c * (2 - c_c) * C) + c_mu * sum_yy
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update step size
            # sigma = sigma * exp(c_sigma / chi_n * (||ps|| / chi_n - 1))
            sigma = sigma * np.exp(c_sigma_eff / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update mean
            x_mean = x_mean_new
            
            # Check for convergence
            # If sigma is very small or variance is very low
            if sigma < 1e-6 or np.max(np.diag(C)) < 1e-12:
                break

        # If we still have budget left, do a local refinement with Nelder-Mead or just random perturbations?
        # Given the directive to switch approach, let's stick to CMA-ES.
        # If budget remains, we can try to refine the best point found by CMA-ES with a few random steps.
        
        if x_best is not None and calls_made < budget:
            # Local random search around best
            for _ in range(budget - calls_made):
                if calls_made >= budget:
                    break
                step = rng.uniform(-0.5, 0.5, dim)
                x_try = np.clip(x_best + step, -5, 5)
                f_try = eval_func(x_try)
                
        return x_best, f_best
# EVOLVE-BLOCK-END
