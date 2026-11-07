"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES with adaptive population sizing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        call_count = 0
        max_calls = self.budget
        best_f = float('inf')
        best_x = np.zeros(self.dim)
        rng = np.random.default_rng(42)
        
        def counted_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= max_calls:
                raise RuntimeError("Budget exceeded")
            call_count += 1
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-ES implementation
        def cma_es(center, sigma, popsize, max_iter):
            nonlocal call_count, best_f, best_x, deadline
            
            dim = self.dim
            # CMA-ES parameters
            mu = popsize // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            
            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)
            step_size = sigma
            
            # Precompute sampling matrix (Cholesky of C)
            # We'll update this each iteration
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                if call_count >= max_calls:
                    break
                
                # Sample population
                L = np.linalg.cholesky(C)
                samples = []
                for i in range(popsize):
                    z = rng.standard_normal(dim)
                    x = center + step_size * (L @ z)
                    samples.append(x)
                
                # Evaluate
                fvals = []
                for x in samples:
                    fvals.append(counted_func(x))
                
                fvals = np.array(fvals)
                
                # Sort by fitness
                idx = np.argsort(fvals)
                samples = [samples[i] for i in idx]
                fvals = fvals[idx]
                
                # Update best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = samples[0].copy()
                
                # Weighted mean
                x_mean = np.zeros(dim)
                for i in range(mu):
                    x_mean += weights[i] * samples[i]
                
                # Update sigma
                z_mean = np.zeros(dim)
                for i in range(mu):
                    z_i = np.linalg.solve(L, (samples[i] - center) / step_size)
                    z_mean += weights[i] * z_i
                
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_mean
                hsig = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) < (1 + 2 / (dim + 1)) * np.sqrt(mu_eff)
                step_size *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                step_size = max(1e-10, min(step_size, 100))
                
                # Update C
                p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * np.linalg.solve(L, (x_mean - center) / step_size)
                
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = np.linalg.solve(L, (samples[i] - center) / step_size)
                    rank_mu += weights[i] * np.outer(y_i, y_i)
                
                C = (1 + c_c * (2 - c_c) - c_sigma) * C + c_sigma * (rank_mu + (1 - hsig) * c_c * (2 - c_c) * np.outer(p_c, p_c))
                
                # Symmetrize
                C = (C + C.T) / 2
                
                center = x_mean
            
            return best_x, best_f

        # Multi-start strategy: 5 restarts with decreasing sigma
        # First restart: large sigma for global exploration
        # Later restarts: smaller sigma for local refinement
        
        # Phase 1: Global exploration with large sigma
        if call_count < max_calls and time.time() < deadline:
            popsize = 20
            center = np.zeros(self.dim)
            sigma = 3.0
            max_iter = 15  # 15 * 20 = 300 calls max
            _, _ = cma_es(center, sigma, popsize, max_iter)
        
        # Phase 2: Local refinement around best found
        if call_count < max_calls and time.time() < deadline:
            remaining = max_calls - call_count
            if remaining > 50:
                popsize = 10
                sigma = 0.5
                max_iter = remaining // popsize
                _, _ = cma_es(best_x, sigma, popsize, max_iter)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
