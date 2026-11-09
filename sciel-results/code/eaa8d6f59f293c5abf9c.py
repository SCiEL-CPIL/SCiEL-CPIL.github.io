"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        popsize = max(4, int(round(4 + 3 * np.log(n))))
        # popsize = max(4, int(round(4 + 3 * np.log(n))))
        
        mu = popsize // 2
        weights = np.log(np.arange(mu, 0, -1)) - np.log(mu - 0.5)
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / (n ** 1.6 + 10)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / (n ** 2 + 10))
        
        # Initial state
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        L = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        
        # Sobol-like initialization: sample a few points to find a good starting point
        # Use deterministic Halton-like sequence for initial exploration
        rng = np.random.default_rng(42)
        
        # Do a quick grid/Sobol search to find a better starting point
        n_init = min(20, budget // 5)
        init_points = []
        for i in range(n_init):
            # Use a low-discrepancy sequence
            point = np.zeros(n)
            for j in range(n):
                # Halton sequence
                m = i + 1
                base = j + 2
                f = 1.0
                r = 0.0
                while m > 0:
                    f /= base
                    r += f * (m % base)
                    m //= base
                point[j] = -5 + 10 * r
            init_points.append(point)
        
        for point in init_points:
            if time.time() > deadline:
                break
            f_val = func(point)
            if f_val < best_f:
                best_f = f_val
                best_x = point.copy()
        
        # Now run CMA-ES from the best initial point
        x_mean = best_x.copy()
        
        count = n_init + 1
        max_iter = min(budget - count, 200)  # Limit iterations
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Generate population
            ys = np.zeros((popsize, n))
            fs = np.zeros(popsize)
            
            for k in range(popsize):
                # Generate sample
                # z ~ N(0, I)
                z = rng.standard_normal(n)
                # Transform: x = x_mean + sigma * B @ (D * z)
                y = x_mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                ys[k] = y
                fs[k] = func(y)
                count += 1
                
                if count >= budget or time.time() > deadline:
                    break
            
            # Sort by fitness
            idx = np.argsort(fs)
            fs_sorted = fs[idx]
            ys_sorted = ys[idx]
            
            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = ys_sorted[0].copy()
            
            # Select top mu
            mu_actual = min(mu, popsize)
            y_selected = ys_sorted[:mu_actual]
            w = weights[:mu_actual]
            
            # Update mean
            x_new_mean = np.sum(w[:, np.newaxis] * y_selected, axis=0)
            
            # Update p_sigma
            v = (x_new_mean - x_mean) / sigma
            z_mean = np.linalg.solve(L, v)
            
            h_sigma = 1.0 if np.linalg.norm(p_sigma) < (1.4 + 2 / (n + 1)) * np.sqrt(
                (1 - (1 - c_sigma) ** 2) * (1 - (1 - c_sigma) ** (2 * mu_eff))) else 0.0
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_mean
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * v
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100)
            
            # Update C
            v_selected = (y_selected - x_mean) / sigma
            rank_mu = np.sum(w[:, np.newaxis] * (v_selected * v_selected.T), axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma ** 2) * c_c * (2 - c_c) * C) + c_mu * rank_mu
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
                B, D = np.linalg.eigh(C)
                D = np.sqrt(np.clip(D, 1e-20, None))
                B = np.real(B)
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset to identity
                C = np.eye(n)
                L = np.eye(n)
                B = np.eye(n)
                D = np.ones(n)
            
            x_mean = x_new_mean
            
            # Check convergence
            if sigma < 1e-8 or np.all(np.abs(x_mean - best_x) < 1e-6):
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
