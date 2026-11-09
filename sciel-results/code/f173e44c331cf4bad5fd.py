"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        pop_size = 4 + int(3 * np.log(dim))  # ~4 for dim=5
        pop_size = max(pop_size, 4)
        
        # Initial center and step size
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size
        
        # Covariance matrix (identity initially)
        C = np.eye(dim)
        
        # Eigen decomposition for sampling
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.sqrt(np.maximum(eigvals, 1e-10))
        
        # Step size adaptation parameters
        mu = pop_size // 2  # Number of best individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Damping
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        steps = 0
        evals = 0
        
        # Initial evaluation
        x_eval = x_mean
        f_eval = func(x_eval)
        evals += 1
        best_x = x_eval.copy()
        best_f = f_eval
        
        # Main loop
        while evals < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (eigvecs @ (eigvals * z))
                ys.append(y)
            
            if not ys:
                break
            
            # Evaluate
            f_vals = []
            for y in ys:
                if time.time() > deadline:
                    break
                fy = func(y)
                f_vals.append(fy)
                evals += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            if not f_vals:
                break
            
            # Sort by fitness
            order = np.argsort(f_vals)
            ys = [ys[i] for i in order]
            f_vals = [f_vals[i] for i in order]
            
            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * ys[i]
            x_new = x_mean + y_w
            
            # Update step size
            y_sigma = np.zeros(dim)
            for i in range(mu):
                y_sigma += weights[i] * (ys[i] - x_mean) / sigma
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Limit sigma
            sigma = np.clip(sigma, 1e-8, 1e8)
            
            # Update covariance
            y_c = np.zeros(dim)
            for i in range(mu):
                y_c += weights[i] * (ys[i] - x_mean) / sigma
            
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            
            C = (1 - c_1 - c_mu) * C + c_1 * (2 - mu_eff) * np.outer(p_c, p_c)
            for i in range(mu):
                C += c_mu * weights[i] * np.outer(ys[i] - x_mean, ys[i] - x_mean) / sigma ** 2
            
            # Update eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.sqrt(np.maximum(eigvals, 1e-10))
            
            # Update mean
            x_mean = x_new
            steps += 1
        
        return best_x, best_f
# EVOLVE-BLOCK-END
