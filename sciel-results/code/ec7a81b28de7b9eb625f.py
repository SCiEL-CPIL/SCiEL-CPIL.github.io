"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-restart strategy for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        global_best_x = None
        global_best_f = float('inf')
        
        rng = np.random.default_rng(0)
        used_budget = 0
        max_restarts = 5
        
        for restart in range(max_restarts):
            if used_budget >= self.budget or time.time() > deadline:
                break
                
            # Initialize CMA-ES with a random start point
            x0 = rng.uniform(-5, 5, self.dim)
            fx0 = func(x0)
            used_budget += 1
            global_best_x = x0
            global_best_f = fx0
            
            if used_budget >= self.budget or time.time() > deadline:
                break

            # CMA-ES Parameters
            n = self.dim
            lam = 4 + int(3 * np.log(n))  # 10 for n=5
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            c1 = 2.0 / ((n + 1.3)**2 + mu_eff)
            cmu = (2.0 * (mu_eff - 2.0 + 1.0/mu_eff)) / ((n + 2.0)**2 + mu_eff)
            cs = (mu_eff + 2.0 + 2.0/(n + mu_eff)) / (n + mu_eff + 5.0)
            ds = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0)/(n + 1.0)) - 1.0) + cs
            cc = (4.0 - mu_eff + mu_eff/n) / (n + 4.0 - mu_eff/n)
            hsig = (1.4 + 2.0/(n + 1.0)) * np.sqrt((mu_eff - 1.0 + 1.0/mu_eff) / (n + 1.0) + 1.0)
            
            x = x0.copy()
            sigma = 1.0
            C = np.eye(n)
            C_sqrt = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            local_best_f = fx0
            local_best_x = x0.copy()
            last_best_f = fx0
            stagnation_counter = 0
            
            while used_budget + lam <= self.budget and time.time() < deadline:
                # Sample
                z = rng.standard_normal((lam, n))
                C_sqrt, _ = np.linalg.cholesky(C)
                y = x + sigma * (C_sqrt @ z.T).T
                y = np.clip(y, -5, 5)
                
                # Evaluate
                fy = np.array([func(yi) for yi in y])
                used_budget += lam
                
                # Sort
                idx = np.argsort(fy)
                y_sorted = y[idx]
                fy_sorted = fy[idx]
                
                # Update global best
                if fy_sorted[0] < global_best_f:
                    global_best_f = fy_sorted[0]
                    global_best_x = y_sorted[0].copy()
                
                # Update local best
                if fy_sorted[0] < local_best_f:
                    local_best_f = fy_sorted[0]
                    local_best_x = y_sorted[0].copy()
                
                # Recombination
                y_w = np.sum(weights[:, np.newaxis] * y_sorted[:mu], axis=0)
                
                # Step size adaptation
                y_sigma = (y_w - x) / sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * y_sigma
                
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2.0/(n + 1.0)) * hsig:
                    sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1.0))
                    
                # Mean update
                x = x + cc * y_w
                
                # Covariance update
                p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mu_eff) * y_w
                delta_h = 1.0 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - cc)**2) < (1.4 + 2.0/(n + 1.0)) * hsig else 0.0
                
                rank_mu = np.outer(y_w, y_w)
                rank_one = np.outer(p_c, p_c)
                
                C = (1 + c1 * delta_h - c1 - cmu) * C + cmu * rank_mu + c1 * delta_h * rank_one
                
                # Check stagnation
                if local_best_f < last_best_f - 1e-10:
                    last_best_f = local_best_f
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    
                if stagnation_counter > 20:
                    break

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
