"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        pop_size = 4 + int(3 * np.log(n))  # ~8 for n=5
        if pop_size > budget // 5:
            pop_size = max(2, budget // 5)
        
        # Initial point
        rng = np.random.default_rng(42)
        x = np.zeros(n)
        f_best = func(x)
        
        # CMA-ES state
        sigma = 2.0  # Initial step size
        C = np.eye(n)  # Covariance matrix
        C_inv = np.eye(n)
        p_sigma = np.zeros(n)  # Evolution path for sigma
        p_c = np.zeros(n)  # Evolution path for C
        c_sigma = (pop_size + 2) / (n + 4)
        c_c = (2 + 2.0 / (n + 1)) / (n + 2 + 2.0 / (n + 1))
        d_sigma = 1 + 2 * max(0, (np.sqrt((pop_size + 1) / (n + 1)) - 1) / (n + 3)) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
        
        # Weights for selection
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Track best solution
        best_x = x.copy()
        best_f = f_best
        calls_made = 1
        
        # Generate initial population
        for gen in range(budget // pop_size):
            # Check deadline
            if time.time() > deadline:
                break
            
            # Sample population
            ys = []
            fs = []
            for _ in range(pop_size):
                # Check deadline
                if time.time() > deadline or calls_made >= budget:
                    break
                # Sample from multivariate normal
                z = rng.standard_normal(n)
                # Transform using C^(1/2)
                # Use Cholesky decomposition
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(n)
                y = x + sigma * L @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                ys.append(y)
                fs.append(fy)
            
            if len(ys) == 0:
                break
            
            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = [ys[i] for i in order]
            fs_sorted = [fs[i] for i in order]
            
            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = ys_sorted[0].copy()
            
            # Weighted mean of best mu solutions
            y_mean = np.zeros(n)
            for i in range(mu):
                y_mean += weights[i] * ys_sorted[i]
            
            # Update step size (sigma)
            # Compute C^{-1/2} (y_mean - x)
            try:
                L_inv = np.linalg.inv(np.linalg.cholesky(C))
            except np.linalg.LinAlgError:
                L_inv = np.eye(n)
            y_mean_centered = y_mean - x
            z = L_inv.T @ y_mean_centered
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)
            
            # Update covariance matrix C
            # Compute C^{-1/2} (y_i - x) for i in best mu
            delta_c = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * L_inv.T @ y_mean_centered
            
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                diff = L_inv.T @ (ys_sorted[i] - x)
                rank_mu += weights[i] * np.outer(diff, diff)
            
            rank_one = np.outer(p_c, p_c)
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) / chi_n if gen > 0 else 1.0
            h_sigma = min(1.0, h_sigma)
            
            C = (1 + c_c * (delta_c - 1) - c_c) * C + c_c * rank_one + \
                c_sigma * (h_sigma * (delta_c - 1) - 1) * rank_mu
            
            # Ensure C is positive definite
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                C_inv = np.eye(n)
            
            # Update mean
            x = y_mean
            x = np.clip(x, -5, 5)
        
        # Use remaining budget for local search around best
        remaining = budget - calls_made
        if remaining > 0 and time.time() < deadline:
            # Simple local refinement
            local_sigma = sigma * 0.1
            for _ in range(min(remaining, 50)):
                if time.time() > deadline or calls_made >= budget:
                    break
                z = rng.standard_normal(n)
                y = best_x + local_sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    local_sigma *= 0.9
                else:
                    local_sigma *= 0.7
        
        return best_x, best_f
# EVOLVE-BLOCK-END
