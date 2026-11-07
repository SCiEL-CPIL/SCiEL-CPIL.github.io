"""Rewritten optimizer for BBOB-style f14 (Different powers) task."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    CMA-ES-like optimizer with adaptive step size and rotation.
    Uses a population-based search with covariance adaptation for robust
    convergence on rotated/shifted landscapes.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initialize
        x = np.zeros(dim)
        # Try a few random starting points to pick the best
        best_x = x.copy()
        best_f = func(best_x)
        
        # Quick random initialization scan (small cost)
        for _ in range(5):
            cand = rng.uniform(-5, 5, dim)
            fc = func(cand)
            if fc < best_f:
                best_f = fc
                best_x = cand.copy()
        
        x = best_x.copy()
        f_best = best_f
        calls_used = 6  # 1 initial + 5 scan

        # CMA-ES parameters
        lambda_pop = 4  # population size
        mu = lambda_pop // 2
        sigma = 1.0
        c_sigma = (mu + 2) / (dim + mu + 5)
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 2 + 1 / mu) / ((dim + 2) ** 2 + mu))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        step_size_sigma = chi_n / np.sqrt(2 * dim * (1 - np.exp(-2 / (dim + 1))))
        step_size_cc = (1 + c_c) * np.sqrt((dim + 1.3) * (c_c + 1 / (dim + 1)))

        # We'll use a simplified CMA-ES that adapts sigma and C
        # But to keep it simple and robust, let's use a diagonal CMA-ES first
        # Actually, let's use a proper CMA-ES but with careful budget management

        # For robustness, let's use a simpler approach: 
        # Adaptive Nelder-Mead or a simple population-based method with restarts
        
        # Let's try a different approach: 
        # 1. Use a simple population-based search with adaptive step size
        # 2. Use multiple restarts if budget allows
        
        # Actually, let's implement a proper but simplified CMA-ES
        
        # Reset for CMA-ES
        x = best_x.copy()
        f_best = best_f
        calls_used = 6
        
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute
        B, D = np.linalg.eigh(C)
        D_inv = 1.0 / np.sqrt(D)
        D_sqrt = np.sqrt(D)
        C_inv_sqrt = B @ np.diag(D_inv) @ B.T
        C_sqrt = B @ np.diag(D_sqrt) @ B.T
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Main loop
        while calls_used < budget:
            # Generate population
            pops = []
            pop_fs = []
            
            for i in range(lambda_pop):
                if calls_used >= budget:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                pops.append(y)
                pop_fs.append(fy)
                calls_used += 1
            
            if not pops:
                break
            
            # Sort by fitness
            indices = np.argsort(pop_fs)
            pops = [pops[i] for i in indices]
            pop_fs = [pop_fs[i] for i in indices]
            
            # Update best
            if pop_fs[0] < f_best:
                f_best = pop_fs[0]
                x = pops[0].copy()
            
            # Update mean
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mu_eff = 1 / np.sum(w ** 2)
            
            new_x = np.zeros(dim)
            for i in range(mu):
                new_x += w[i] * pops[i]
            
            # Update path covariance
            y_mean = (new_x - x) / sigma if sigma > 1e-15 else np.zeros(dim)
            pc = (1 - c_c) * pc + c_c * y_mean
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ y_mean
            
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10)
            
            # Update covariance
            delta_sigma = (1 + 2 * max(0, (np.linalg.norm(ps) / chi_n - 1) / (dim + 3))) * c_sigma
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                zi = C_inv_sqrt @ (pops[i] - x) / sigma if sigma > 1e-15 else np.zeros(dim)
                rank_mu += w[i] * np.outer(zi, zi)
            
            C = (1 + c_1 * delta_sigma - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                B, D = np.linalg.eigh(C)
                D = np.maximum(D, 1e-12)
                D_inv = 1.0 / np.sqrt(D)
                D_sqrt = np.sqrt(D)
                C_inv_sqrt = B @ np.diag(D_inv) @ B.T
                C_sqrt = B @ np.diag(D_sqrt) @ B.T
            except:
                C = np.eye(dim)
                B, D = np.linalg.eigh(C)
                D_inv = 1.0 / np.sqrt(D)
                D_sqrt = np.sqrt(D)
                C_inv_sqrt = B @ np.diag(D_inv) @ B.T
                C_sqrt = B @ np.diag(D_sqrt) @ B.T
            
            x = new_x
            x = np.clip(x, -5, 5)
            
            # If stuck, try a restart
            if calls_used > budget * 0.5 and sigma < 0.01:
                # Try a few random restarts
                for _ in range(3):
                    if calls_used >= budget:
                        break
                    cand = rng.uniform(-5, 5, dim)
                    fc = func(cand)
                    calls_used += 1
                    if fc < f_best:
                        f_best = fc
                        x = cand.copy()
                        sigma = 1.0
                        C = np.eye(dim)
                        pc = np.zeros(dim)
                        ps = np.zeros(dim)
                        B, D = np.linalg.eigh(C)
                        D_inv = 1.0 / np.sqrt(D)
                        D_sqrt = np.sqrt(D)
                        C_inv_sqrt = B @ np.diag(D_inv) @ B.T
                        C_sqrt = B @ np.diag(D_sqrt) @ B.T
        
        return x, f_best
# EVOLVE-BLOCK-END
