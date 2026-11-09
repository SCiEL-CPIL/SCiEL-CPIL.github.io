"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        
        # Use a population-based evolutionary strategy (CMA-ES simplified)
        # with adaptive covariance and step size
        
        n = self.dim
        budget = self.budget
        
        # Initial population size
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        
        # We'll use a simplified CMA-ES
        # Initialize parameters
        x = np.zeros(n)  # Start at center
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Adaptation parameters
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # Track best
        best_x = None
        best_f = np.inf
        
        # Initial evaluation
        x = rng.uniform(-3, 3, n)
        fx = func(x)
        best_x, best_f = x.copy(), fx
        
        calls_used = 1
        remaining = budget - calls_used
        
        # Main loop
        while remaining > 0:
            # Generate population
            offspring = []
            offspring_f = []
            
            # Number of offspring to generate in this iteration
            n_gen = min(lambda_, remaining)
            
            for i in range(n_gen):
                z = rng.standard_normal(n)
                # Transform using Cholesky of C
                # For simplicity, use diagonal approximation initially, 
                # but let's do proper CMA-ES
                B, D = np.linalg.eigh(C)
                B = B[:, B > 1e-10]
                D = D[D > 1e-10]
                D = np.sqrt(D)
                y = B @ (D * z)
                x_new = x + sigma * y
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                offspring.append(x_new)
                offspring_f.append(f_new)
                remaining -= 1
                calls_used += 1
                
                if f_new < best_f:
                    best_x, best_f = x_new.copy(), f_new
            
            if remaining <= 0:
                break
            
            # Sort offspring by fitness
            offspring = np.array(offspring)
            offspring_f = np.array(offspring_f)
            order = np.argsort(offspring_f)
            offspring = offspring[order]
            offspring_f = offspring_f[order]
            
            # Update mean
            x_new = np.sum(weights[:n_gen] * offspring[:n_gen], axis=0)
            
            # Update path
            y_sigma = (x_new - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**min(1, calls_used / (n + 1)))) <= (1 + 2 / (n + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * y_sigma
            
            # Update covariance
            delta_d = (1 + hsig * (1 / c_c - 1) * (1 - c_c)) * c_mu * mueff
            C = (1 + c_1 + c_mu - c_1 - c_mu) * C + \
                c_1 * (pc @ pc.T + (1 - hsig) * (c_c * (2 - c_c) * np.eye(n))) + \
                c_mu * np.sum([weights[i] * np.outer(offspring[i] - x, offspring[i] - x) for i in range(n_gen)], axis=0)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            x = x_new
            
            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Reset if sigma is too large or too small
            if sigma > 10:
                sigma = 10
            if sigma < 1e-8:
                sigma = 1e-8
        
        return best_x, best_f
# EVOLVE-BLOCK-END
