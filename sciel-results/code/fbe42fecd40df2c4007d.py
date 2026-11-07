"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with deadline guard for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill
        budget = self.budget
        dim = self.dim

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def safe_call(x):
            nonlocal calls
            if calls >= budget:
                return None
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            return fx

        def check_time():
            return time.time() >= deadline

        # Multi-start CMA-ES: several short CMA-ES runs from different starting points
        # This helps escape local minima in the 101-peak landscape
        
        # Use a set of diverse starting points
        rng = np.random.default_rng(42)
        
        # Generate 5 diverse starting points using Latin Hypercube-like spread
        n_starts = 5
        starts = []
        for i in range(n_starts):
            # Spread across the domain with some randomness
            x0 = rng.uniform(-4, 4, dim)
            starts.append(x0)
        
        # Also add center
        starts.append(np.zeros(dim))
        
        # Allocate budget among starts: first start gets more, others less
        # CMA-ES needs at least ~10*dim evaluations to be useful
        min_per_start = max(30, 5 * dim)
        
        # Try each start with CMA-ES
        for idx, x0 in enumerate(starts):
            if check_time():
                break
            if calls >= budget:
                break
            
            # Determine budget for this start
            remaining = budget - calls
            if idx == 0:
                # First start gets a larger share
                alloc = min(remaining, max(200, budget // 2))
            else:
                # Subsequent starts get smaller shares
                alloc = min(remaining, max(min_per_start, 100))
            
            if alloc < min_per_start:
                break
            
            fx = safe_call(x0)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()
            
            # Run CMA-ES from this starting point
            # CMA-ES parameters
            pop_size = 4 + int(3 * np.log(dim))  # ~10 for dim=5
            sigma = 1.5
            x_mean = x0.copy()
            C = np.eye(dim)
            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # CMA-ES update parameters
            c_sigma = (pop_size + 2) / (dim + 4)
            mu = pop_size // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w**2)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2)**2 + mu_eff))
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))
            
            # Generate initial population
            for gen in range(alloc):
                if check_time() or calls >= budget:
                    break
                
                # Generate offspring
                ys = np.zeros((pop_size, dim))
                fvals = np.zeros(pop_size)
                
                for j in range(pop_size):
                    if check_time() or calls >= budget:
                        break
                    # Sample from N(x_mean, sigma^2 * C)
                    z = rng.standard_normal(dim)
                    # Cholesky decomposition of C
                    try:
                        L = np.linalg.cholesky(C)
                    except:
                        L = np.eye(dim)
                    y = x_mean + sigma * L @ z
                    fy = safe_call(y)
                    if fy is None:
                        break
                    ys[j] = y
                    fvals[j] = fy
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                
                # Sort by fitness
                order = np.argsort(fvals)
                ys = ys[order]
                fvals = fvals[order]
                
                # Update mean
                x_new_mean = np.sum(w[:, np.newaxis] * ys[:mu], axis=0)
                
                # Update covariance
                yw = (ys[:mu] - x_mean) / sigma
                C_new = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - c_c) * (1 - c_c) * C)
                C_new += c_mu * np.sum(w[:, np.newaxis, np.newaxis] * np.outer(yw, yw), axis=0)
                # Ensure symmetry and positive definite
                C_new = (C_new + C_new.T) / 2
                C = C_new
                
                # Update evolution paths
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L.T @ (x_new_mean - x_mean) / sigma)
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (L.T @ (x_new_mean - x_mean) / sigma)
                
                # Sigma update
                sigma *= np.exp((c_sigma / chi_n) * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 100)
                
                x_mean = x_new_mean
        
        return best_x, best_f
# EVOLVE-BLOCK-END
