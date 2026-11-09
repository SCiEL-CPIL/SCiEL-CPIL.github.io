"""Rewritten optimizer for BBOB f20 (Schwefel) using CMA-ES with global restarts."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with global restarts for multimodal search on Schwefel-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.asarray(x, dtype=np.float64).flatten()
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                return np.inf
            if not np.isfinite(fx):
                fx = np.inf
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global coverage using stratified sampling
        n_global = 50
        
        rng = np.random.default_rng(42)
        n_per_dim = max(2, int(round(n_global ** (1.0 / dim))))
        
        from itertools import product
        grid_indices = list(product(range(n_per_dim), repeat=dim))
        rng.shuffle(grid_indices)
        
        global_points = []
        for idx in grid_indices[:n_global]:
            point = np.zeros(dim)
            for d in range(dim):
                lo = -5.0 + 10.0 * idx[d] / n_per_dim
                hi = -5.0 + 10.0 * (idx[d] + 1) / n_per_dim
                point[d] = rng.uniform(lo, hi)
            global_points.append(point)
        
        for x in global_points:
            if time.time() > deadline or call_count >= budget:
                break
            safe_call(x)

        remaining_budget = budget - call_count
        if remaining_budget <= 0:
            return best_x, best_f

        # Phase 2: CMA-ES with restarts
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))  # ~14 for dim=5
        
        # CMA-ES parameters
        c1 = (4 + np.log(dim)) / ((dim + 4) * (3 * dim + 9))
        cmu = min(1 - c1, 2 * (2 - np.sqrt(dim + 2) + 1) / ((dim + 2) * (3 * dim + 9)))
        c_sigma = (mu_eff + 2) / ((dim + 8) * (mu_eff / 2 + 5))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        
        mu_eff = lambda_ / 2
        weights = np.log(mu_eff + 0.5) - np.log(np.arange(1, mu_eff + 1))
        weights /= weights.sum()
        
        # Initial state
        mean = best_x.copy()
        sigma = 1.5
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Restart tracking
        max_stagnant = 60
        stagnant_count = 0
        current_best_f = best_f
        
        # CMA-ES loop
        while call_count < budget and time.time() < deadline:
            # Generate population
            pop = []
            for i in range(lambda_):
                # Sample from N(mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                x_i = mean + sigma * (C_inv_sqrt @ z)
                x_i = np.clip(x_i, -5.0, 5.0)
                pop.append(x_i)
            
            # Evaluate population
            pop_f = np.array([safe_call(x) for x in pop])
            
            # Sort by fitness
            order = np.argsort(pop_f)
            sorted_pop = [pop[i] for i in order]
            sorted_f = pop_f[order]
            
            # Update best
            if sorted_f[0] < current_best_f:
                current_best_f = sorted_f[0]
                stagnant_count = 0
            else:
                stagnant_count += 1
            
            # Check for stagnation and restart
            if stagnant_count >= max_stagnant:
                # Try random restarts
                restarts_found = False
                for _ in range(3):
                    if call_count >= budget or time.time() > deadline:
                        break
                    restart_x = rng.uniform(-5.0, 5.0, size=dim)
                    restart_f = safe_call(restart_x)
                    if np.isfinite(restart_f) and restart_f < current_best_f:
                        mean = restart_x.copy()
                        current_best_f = restart_f
                        sigma = 1.5
                        C = np.eye(dim)
                        C_inv_sqrt = np.eye(dim)
                        pc = np.zeros(dim)
                        ps = np.zeros(dim)
                        stagnant_count = 0
                        restarts_found = True
                        break
                if not restarts_found:
                    # Reduce sigma and continue
                    sigma *= 0.7
                    stagnant_count = 0
                    if sigma < 0.01:
                        break
            
            # Update mean (weighted recombination)
            old_mean = mean.copy()
            mean = sum(weights[i] * sorted_pop[i] for i in range(mu_eff))
            
            # Update covariance matrix
            diff = mean - old_mean
            diff = diff / sigma if sigma > 0 else diff
            C += cmu * sum(weights[i] * np.outer(sorted_pop[i] - old_mean, sorted_pop[i] - old_mean) / (sigma ** 2) for i in range(mu_eff))
            
            # Update step size (sigma)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * diff
            h_sig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update evolution path for covariance
            pc = (1 - c1) * pc + h_sig * np.sqrt(c1 * (2 - c1) * mu_eff) * diff
            rank_one = np.outer(pc, pc)
            C += (1 - (1 - c1) * (1 - cmu) - c1 * c1 * mu_eff / 2) * (rank_one - C)
            
            # Ensure C is positive definite and update C_inv_sqrt
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = (eigvecs * eigvals) @ eigvecs.T
                C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
            except Exception:
                C = np.eye(dim)
                C_inv_sqrt = np.eye(dim)

        # Final: ensure we have the global best
        if current_best_f < best_f:
            best_f = current_best_f
            best_x = mean.copy()

        return best_x, best_f
