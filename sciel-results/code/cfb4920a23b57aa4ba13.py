"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer: brief diversification followed by lightweight CMA-ES.
    
    Strategy:
    1. Initial Diversification: Sample diverse points to find a promising basin.
    2. Lightweight CMA-ES: Adapt both step size and covariance to navigate
       correlated/elongated landscapes better than isotropic (1+1)-ES.
    3. Safety: Strict deadline and budget enforcement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        rng = np.random.default_rng(42)
        n = self.dim
        total_budget = self.budget
        calls_made = 0

        best_x = None
        best_f = np.inf

        # Phase 1: Initial Diversification
        # Sample origin + 4 random points = 5 calls
        n_init_samples = 4
        init_candidates = [np.zeros(n)]
        for _ in range(n_init_samples):
            init_candidates.append(rng.uniform(-5, 5, n))

        for x in init_candidates:
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made = 1

        # Phase 2: Lightweight CMA-ES
        x = best_x.copy()
        f = best_f
        
        # CMA-ES parameters for dim=5
        mu = 2  # number of parents
        lambda_ = 5  # population size
        c_sigma = (mu + 0.5) / (n + mu + 5)
        c_c = (4 + mu / n) / (n + 4 + 2 * mu / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * ((mu - 2 + 1 / mu) / ((n + 2) ** 2 + mu)))
        
        # Initialize CMA-ES state
        sigma = 1.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Precompute weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)
        
        remaining_budget = total_budget - calls_made
        
        while remaining_budget > 0 and time.time() < deadline:
            # Generate population
            offspring = []
            offspring_f = []
            
            for i in range(lambda_):
                if calls_made >= total_budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Sample from N(x, sigma^2 * C)
                z = rng.standard_normal(n)
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                except np.linalg.LinAlgError:
                    L = np.eye(n)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                
                offspring.append(y)
                fy = func(y)
                calls_made += 1
                offspring_f.append(fy)
                remaining_budget -= 1
            
            if not offspring:
                break
            
            # Sort by fitness
            indices = np.argsort(offspring_f)
            offspring = [offspring[i] for i in indices]
            offspring_f = [offspring_f[i] for i in indices]
            
            # Take best mu offspring
            offspring_mu = offspring[:mu]
            
            # Compute weighted mean
            new_mean = np.zeros(n)
            for i in range(mu):
                new_mean += weights[i] * offspring_mu[i]
            
            # Update path cumulant
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C @ (new_mean - x) / sigma
            norm_ps = np.linalg.norm(ps)
            
            # Update path evolution strategy
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - x) / sigma
            
            # Update sigma
            sigma *= np.exp(c_sigma * (norm_ps / np.sqrt(2 * n) - 1))
            sigma = max(0.01, min(sigma, 10.0))
            
            # Update covariance matrix
            delta_D = (1 + 2 * (min(1, (1 - c_c) * (np.dot(pc, pc) / n) - 1)) * c_1 / c_mu - 1)
            C = (1 - c_1 - c_mu + c_1 * delta_D) * C + c_1 * np.outer(pc, pc)
            for i in range(mu):
                diff = offspring_mu[i] - new_mean
                C += c_mu * weights[i] * np.outer(diff, diff)
            
            # Ensure C is symmetric and positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update best
            if offspring_f[0] < best_f:
                best_f = offspring_f[0]
                best_x = offspring[0].copy()
            
            x = new_mean
            f = offspring_f[0]
            
            # Check for convergence
            if np.linalg.norm(new_mean - x) < 1e-10 and sigma < 0.01:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
