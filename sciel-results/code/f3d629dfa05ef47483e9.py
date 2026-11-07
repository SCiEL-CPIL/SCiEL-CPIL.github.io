"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(0)
        
        n = self.dim
        lambda_ = 4 + int(3 * np.log(n))
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (n + 2) - 1))
        
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        best_x = x_mean.copy()
        best_f = func(x_mean)
        calls_used = 1
        
        while calls_used + lambda_ <= self.budget:
            if time.time() > deadline:
                break
                
            xs = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    sqrt_eigvals = np.sqrt(eigvals)
                    z_rot = eigvecs @ (sqrt_eigvals * (eigvecs.T @ z))
                    y = x_mean + sigma * z_rot
                except np.linalg.LinAlgError:
                    y = x_mean + sigma * z
                    
                y = np.clip(y, -5, 5)
                xs.append(y)
            
            fs = np.array([func(x) for x in xs])
            calls_used += lambda_
            
            idx_min = np.argmin(fs)
            if fs[idx_min] < best_f:
                best_f = fs[idx_min]
                best_x = xs[idx_min].copy()
            
            order = np.argsort(fs)
            y_sorted = np.array([xs[i] for i in order])
            y_w = y_sorted[:mu]
            
            x_new = np.sum(w[:, None] * y_w, axis=0)
            
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                inv_sqrt_eigvals = 1.0 / np.sqrt(eigvals)
                C_inv_sqrt = eigvecs @ (inv_sqrt_eigvals * (eigvecs.T))
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(n)
                
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (x_new - x_mean) / sigma)
            
            denom = np.sqrt(1 - (1 - c_sigma) ** (2 * calls_used))
            if denom < 1e-8:
                h_sigma = 0
            else:
                h_sigma = 1 if np.linalg.norm(ps) / denom < (1.4 + 2 / (n + 1)) * np.sqrt(n) else 0
                
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            
            pc = (1 - c_c) * pc + h_sigma * c_c * (x_new - x_mean) / sigma
            
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum(w[:, None, None] * (y_w - x_mean)[:, :, None] * (y_w - x_mean)[:, None, :], axis=0) / (sigma ** 2)
            
            C = (1 + c_c * (2 - c_c) * h_sigma - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            C = (C + C.T) / 2
            C += 1e-6 * np.eye(n)
            
            x_mean = x_new
            
        return best_x, best_f
# EVOLVE-BLOCK-END
