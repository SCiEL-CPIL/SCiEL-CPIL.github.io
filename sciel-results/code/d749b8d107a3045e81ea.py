"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES-like optimizer with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget
        
        # Initial population size
        lam = max(4, int(4 + 3 * np.log(n)))
        
        # Initialize at a random point
        x = rng.uniform(-3, 3, n)
        f_best = func(x)
        
        # CMA-ES parameters
        sigma = 1.0
        c1 = (2 / ((n + 1.3)**2 + lam))
        cmu = min(0.4 - c1, 2 / ((n + 2)**2 + lam))
        cs = (lam + 2) / (n + lam + 5)
        c1 = (2 / ((n + 1.3)**2 + lam))
        cmu = min(0.4 - c1, 2 / ((n + 2)**2 + lam))
        ds = 1 + 2 * max(0, np.sqrt((lam - 1) / (n + 1)) - 1) + cs
        
        # Covariance matrix
        C = np.eye(n)
        sqrtC = np.eye(n)
        
        # Evolution path
        ps = np.zeros(n)
        hsig = (1 + 2 / (n + 1)) * (1 - cs)
        
        # Initial weights
        weights = np.ones(lam)
        weights[:lam // 2] *= 2
        weights /= weights.sum()
        mueff = 1 / weights.sum()
        alpha_mu = 2 * (mueff - 2 + 1 / mueff) / ((n + mueff) * (mueff - 2))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # Main loop
        calls_used = 1
        stall_count = 0
        f_prev_best = f_best
        
        while calls_used < budget:
            # Generate offspring
            offspring = []
            f_offspring = []
            
            for i in range(lam):
                z = rng.standard_normal(n)
                y = x + sigma * (sqrtC @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)
                
                if calls_used >= budget:
                    break
                f_offspring.append(func(y))
                calls_used += 1
            
            if len(f_offspring) == 0:
                break
            
            # Sort offspring
            idx = np.argsort(f_offspring)
            f_offspring = [f_offspring[i] for i in idx]
            offspring = [offspring[i] for i in idx]
            
            # Update best
            if f_offspring[0] < f_best:
                f_best = f_offspring[0]
                stall_count = 0
            else:
                stall_count += 1
            
            # Update mean
            x_new = x + sum(weights[i] * (offspring[i] - x) for i in range(min(mueff, len(offspring))))
            
            # Update evolution path
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi_n * (sqrtC.T @ (x_new - x) / sigma)
            
            # Check for restart
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * 1)) > 1.4 + 2 / (n + 1):
                hsig = 0
            else:
                hsig = (1 + 2 / (n + 1)) * (1 - cs)
            
            # Update covariance
            delta_h = (1 - hsig) * cs * (2 - cs)
            rank_one = np.outer(ps, ps)
            rank_mue = sum(weights[i] * np.outer(offspring[i] - x, offspring[i] - x) for i in range(min(mueff, len(offspring))))
            C = (1 + cs * ds * delta_h - c1 - cmu) * C + c1 * rank_one + cmu * rank_mue
            
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update sqrtC using Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
                sqrtC = L
            except np.linalg.LinAlgError:
                # Regularize and retry
                C += np.eye(n) * 1e-10
                try:
                    L = np.linalg.cholesky(C)
                    sqrtC = L
                except np.linalg.LinAlgError:
                    sqrtC = np.eye(n)
                    C = np.eye(n)
                    break
            
            x = x_new
            
            # Early stopping if no improvement for a while
            if stall_count > 30 and calls_used > budget * 0.3:
                break
        
        return x, f_best
# EVOLVE-BLOCK-END
