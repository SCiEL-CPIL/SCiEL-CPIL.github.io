"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation for sharp ridge optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s kill

        def safe_call(x):
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            try:
                return float(func(x))
            except Exception:
                return np.inf

        # Global bookkeeping
        best_x = None
        best_f = np.inf
        calls_made = 0

        def track_call(x, f):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # 1. Global sampling to find a good starting point
        n_global = 15
        points = []
        rng = np.random.default_rng(42)
        for _ in range(n_global):
            x = rng.uniform(-4, 4, dim)
            f = safe_call(x)
            track_call(x, f)
            points.append((f, x))
            
        points.sort(key=lambda t: t[0])

        # 2. CMA-ES local optimization from the best point
        if calls_made < budget and time.time() < deadline:
            x0 = points[0][1].copy()
            
            # CMA-ES parameters
            lambda_pop = 4 * dim + 2  # population size
            mu = lambda_pop // 2  # number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            sigma = 2.0  # initial step size
            mean = x0.copy()
            C = np.eye(dim)  # covariance matrix
            C_inv_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Precompute square root of C for sampling
            def update_covariance_sqrt():
                nonlocal C, C_inv_sqrt
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except:
                    C_inv_sqrt = np.eye(dim)
            
            update_covariance_sqrt()
            
            # Main CMA-ES loop
            gen = 0
            while calls_made < budget and time.time() < deadline:
                gen += 1
                # Sample population
                pop = []
                for i in range(lambda_pop):
                    z = rng.standard_normal(dim)
                    x = mean + sigma * (C_inv_sqrt @ z)
                    x = np.clip(x, -5, 5)
                    f = safe_call(x)
                    track_call(x, f)
                    pop.append((f, x))
                
                if calls_made >= budget or time.time() > deadline:
                    break
                    
                # Sort by fitness
                pop.sort(key=lambda t: t[0])
                
                # Update mean
                new_mean = np.zeros(dim)
                for i in range(mu):
                    new_mean += weights[i] * pop[i][1]
                
                # Update sigma
                old_mean = mean.copy()
                mean = new_mean
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_inv_sqrt @ (mean - old_mean))
                if np.linalg.norm(p_sigma) / chi_n > 1.4 + 2 / (dim + 1):
                    sigma *= np.exp(c_sigma / 5 * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update covariance
                diff = (mean - old_mean) / sigma
                delta_sig = 1 if np.linalg.norm(p_sigma) / chi_n > 1.4 + 2 / (dim + 1) else 0
                p_c = (1 - c_c) * p_c + c_c * diff
                rank_1 = np.outer(p_c, p_c)
                
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    diff_i = (pop[i][1] - old_mean) / sigma
                    rank_mu += weights[i] * np.outer(diff_i, diff_i)
                
                C = (1 + c_1 * (2 - c_1) + c_mu - c_1) * C + c_1 * rank_1 + c_mu * rank_mu
                
                # Ensure positive definiteness
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                update_covariance_sqrt()
                
                # Adapt sigma based on progress
                if gen % 10 == 0 and mu > 0:
                    progress = abs(pop[0][0] - pop[-1][0])
                    if progress < 1e-6:
                        sigma *= 0.5
                    elif progress > 1:
                        sigma *= 1.1
                
                # Stop if sigma is too small or too large
                if sigma < 1e-8 or sigma > 100:
                    break
                    
        return best_x, best_f
# EVOLVE-BLOCK-END
