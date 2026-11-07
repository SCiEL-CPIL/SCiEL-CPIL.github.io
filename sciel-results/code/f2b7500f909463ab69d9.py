"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 320  # Safety margin before 330s kill
        n = self.dim
        B = self.budget

        # Initialize CMA-ES parameters
        # Population size
        pop_size = max(4, int(4 + 3 * np.log(n)))
        
        # Initial point: center of the box
        x_mean = np.zeros(n)
        
        # Initial step size: 25% of the box width
        sigma = 2.5
        
        # Bounds
        lower = -5.0 * np.ones(n)
        upper = 5.0 * np.ones(n)
        
        # Data
        best_x = x_mean.copy()
        best_f = np.inf
        calls = 0
        
        def add_point(x, f):
            nonlocal best_x, best_f, calls
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # CMA-ES Parameters
        # Learning rates
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        c1 = (2 - mu_eff / (mu_eff + 1)) / (n + 1) + 2 * max(0, c_sigma - 1 / (n + 1))
        cmu = (2 - c1) * min(1, (mu_eff - 2 + 1 / mu_eff) / (n + 2))
        
        # State variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        C_sqrt = np.eye(n)
        
        def compute_inv_sqrt(C):
            # Compute inverse square root of C
            # Using eigenvalue decomposition
            vals, vecs = np.linalg.eigh(C)
            vals = np.clip(vals, 1e-10, None)
            C_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
            return C_inv_sqrt
        
        C_inv_sqrt = compute_inv_sqrt(C)
        
        while calls < B:
            if time.time() - start_time > time_limit:
                break
            
            # Generate population
            X = np.zeros((pop_size, n))
            for i in range(pop_size):
                z = np.random.randn(n)
                X[i] = x_mean + sigma * (C_sqrt @ z)
                # Clip to bounds
                X[i] = np.clip(X[i], lower, upper)
            
            # Evaluate population
            F = np.zeros(pop_size)
            for i in range(pop_size):
                if calls >= B:
                    break
                F[i] = func(X[i])
                add_point(X[i], F[i])
            
            if calls >= B:
                break
            
            # Sort by fitness
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # Update best
            # (Already updated in add_point, but ensure best_x is from this pop if better)
            # Note: add_point updates global best.
            
            # Weighted mean
            x_new_mean = np.sum(weights[:, np.newaxis] * X[:mu], axis=0)
            
            # Update step size (sigma)
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ ((x_new_mean - x_mean) / sigma))
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(n) * (1 - 1 / (2 * n**2) + 1 / (8 * n**4))
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # Update covariance matrix
            # p_c update
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * ((x_new_mean - x_mean) / sigma)
            
            # Rank-mu update
            v = (x_new_mean - x_mean) / sigma
            rank_mu = np.sum(weights[:, np.newaxis] * (X[:mu] - x_mean)[:, np.newaxis, :] * (X[:mu] - x_mean)[:, np.newaxis, :].T, axis=0)
            
            # Update C
            C = (1 + c1 + cmu - c1 - cmu) * C + c1 * np.outer(v, v) + cmu * rank_mu
            # Note: The term (1 + c1 + cmu - c1 - cmu) is just 1.
            # Actually, standard update is:
            # C = (1 - c1 - cmu) * C + c1 * v v^T + cmu * rank_mu
            C = (1 - c1 - cmu) * C + c1 * np.outer(v, v) + cmu * rank_mu
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update C_sqrt (approximation using eigendecomposition)
            # Recompute C_inv_sqrt
            C_inv_sqrt = compute_inv_sqrt(C)
            
            # Compute C_sqrt
            # C_sqrt = C_inv_sqrt.T @ C
            # But we need a matrix such that C_sqrt @ C_sqrt.T = C
            # Using eigendecomposition
            vals, vecs = np.linalg.eigh(C)
            vals = np.clip(vals, 1e-10, None)
            C_sqrt = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
            
            # Update mean
            x_mean = x_new_mean
            
            # Clip mean to bounds
            x_mean = np.clip(x_mean, lower, upper)
            
            # Optional: Local refinement if stuck?
            # For now, just keep going.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
