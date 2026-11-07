import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0
        
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        lam = 14
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        chi_n = np.sqrt(n) * (1 + 1 / (2 * n) - 1 / (8 * n ** 2))
        
        # State
        x_mean = np.zeros(n)
        sigma = 1.0
        C = np.eye(n)
        C_sqrt = np.eye(n)
        C_inv_sqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1
        
        gen = 0
        last_improvement_gen = 0
        
        while calls < budget:
            if time.time() > deadline - 5.0:
                break
            
            # Check for stagnation/restart
            if (gen - last_improvement_gen > 10) or (sigma < 1e-4):
                # Restart from best point with larger sigma
                x_mean = best_x.copy()
                sigma = 1.0
                C = np.eye(n)
                C_sqrt = np.eye(n)
                C_inv_sqrt = np.eye(n)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                gen = 0 # Reset gen counter for stagnation check
                last_improvement_gen = 0
                # We don't reset calls or best_f
                
            ys = []
            fs = []
            
            for i in range(lam):
                if calls >= budget:
                    break
                z = rng.standard_normal(n)
                y = x_mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                ys.append(y)
                fs.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    last_improvement_gen = gen
            
            if len(ys) < mu:
                break
            
            ys = np.array(ys)
            fs = np.array(fs)
            
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]
            
            y_w = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)
            d = y_w - x_mean
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ d) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * d
            
            # h_sigma calculation
            # norm(p_sigma) / sqrt(1 - (1-c_sigma)^(2*(gen+1)))
            # Note: gen is reset on restart, so this is correct for the current run
            denom = np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / denom < (1.4 + 2 / (n + 1)) * chi_n) else 0.0
            
            C = ((1 + c_1 * h_sigma * (h_sigma - 1) - c_1 - c_mu) * C +
                 c_1 * np.outer(p_c, p_c) +
                 c_mu * np.sum(weights[:, None, None] * np.outer(ys_sorted[:mu] - x_mean, ys_sorted[:mu] - x_mean), axis=0))
            
            C = (C + C.T) / 2.0
            
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(1e-10, min(sigma, 10.0))
            
            x_mean = y_w
            x_mean = np.clip(x_mean, -5, 5)
            
            gen += 1
        
        return best_x, best_f
