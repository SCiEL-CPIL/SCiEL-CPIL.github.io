"""Rewritten optimizer for BBOB f14 (Different powers) using CMA-ES hybrid."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid CMA-ES and (1+1)-ES optimizer for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s kill
        rng = np.random.default_rng(42)
        d = self.dim
        max_calls = self.budget

        best_x = None
        best_f = float('inf')
        calls_used = 0

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            if time.time() > deadline - 2:
                return None
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start to find a good initial basin
        # Use diverse starting points: origin and random points
        starts = [np.zeros(d)]
        for _ in range(5):
            starts.append(rng.uniform(-5, 5, d))

        for s in starts:
            s_clipped = np.clip(s, -5, 5)
            fx = try_eval(s_clipped)
            if fx is None:
                break

        if calls_used >= max_calls or time.time() > deadline - 10:
            return best_x, best_f

        # Phase 2: CMA-ES for global/mid-range optimization
        # CMA-ES is better at handling anisotropic landscapes like f14
        x0 = best_x.copy()
        sigma0 = 2.0  # Initial step size for CMA-ES
        max_cma_calls = int(max_calls * 0.7)  # Use 70% of budget for CMA-ES
        
        # CMA-ES Parameters
        lam = max(4, int(4 + 3 * np.log(d)))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_cov_pos = (2 + 2 * mu_eff / d) / (d + 2 + mu_eff / d)
        c_cov_neg = min(1 - c_cov_pos, 2 * (mu_eff - 2 + 1.0 / mu_eff) / (d ** 2 + d))
        
        x_mean = x0.copy()
        sigma = sigma0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        chi_n = np.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (20 * d ** 2))

        cma_calls_spent = 0
        while cma_calls_spent < max_cma_calls and calls_used < max_calls:
            if time.time() > deadline - 5:
                break
            
            # Generate population
            pop = np.zeros((lam, d))
            pop_f = np.zeros(lam)
            valid_count = 0
            
            for i in range(lam):
                if calls_used >= max_calls or time.time() > deadline - 3:
                    break
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(d)
                
                z = rng.standard_normal(d)
                x_new = x_mean + sigma * (L @ z)
                x_new = np.clip(x_new, -5, 5)
                
                fx = try_eval(x_new)
                if fx is None:
                    break
                
                pop[i] = x_new
                pop_f[i] = fx
                valid_count += 1
                cma_calls_spent += 1
            
            if valid_count < 2:
                break

            # Sort by fitness
            order = np.argsort(pop_f[:valid_count])
            pop = pop[order]
            pop_f = pop_f[order]

            # Update best
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = pop[0].copy()

            # Weighted mean
            x_new_mean = np.sum(weights[:valid_count//2 if valid_count//2 < mu else mu][:min(mu, valid_count)], None) * 0 + \
                         np.sum(weights[:min(mu, valid_count)][:, None] * pop[:min(mu, valid_count)], axis=0)
            
            # Update step size control vector
            # Note: L should ideally be the inverse Cholesky or related to the covariance update
            # For simplicity and stability in this short horizon, we use the standard update
            try:
                L_inv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                L_inv = np.eye(d)
                
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (L_inv @ (x_new_mean - x_mean))
            norm_ps = np.linalg.norm(ps)

            # Update evolution path
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma

            # Update covariance matrix
            delta = 0 if norm_ps ** 2 / chi_n ** 2 < 2 * (1 + 1 / c_c) else 1
            C = (1 + c_cov_pos + c_cov_neg - c_c) * C + \
                c_cov_pos * np.outer(pc, pc) + \
                c_cov_neg * (1 - delta) * np.outer(ps, ps) - \
                c_cov_neg * (1 - delta) * chi_n ** 2 * C

            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C += 1e-10 * np.eye(d)

            # Update mean and sigma
            x_mean = x_new_mean
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            
            # Safety cap for sigma
            if sigma > 10.0:
                sigma = 10.0
            if sigma < 1e-8:
                break

        # Phase 3: Polish phase with (1+1)-ES
        # Use the current best point and a small sigma to fine-tune
        if calls_used < max_calls and time.time() < deadline - 2:
            x = best_x.copy()
            fx = best_f
            # Start with a small sigma since CMA-ES should have converged somewhat
            sigma = max(0.1, sigma * 0.5) 
            polish_calls = max_calls - calls_used
            
            for _ in range(polish_calls):
                if time.time() > deadline - 1:
                    break
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05  # Slight increase on success
                else:
                    sigma *= 0.9   # Decrease on failure
                    
                # If sigma becomes too small, we can break early or continue
                if sigma < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
