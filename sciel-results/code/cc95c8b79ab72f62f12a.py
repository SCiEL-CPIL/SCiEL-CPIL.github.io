"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = 8  # population size
        mu = lam // 2  # number of selected parents
        # Logarithmic weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        def cma_es(x0, sigma0, remaining_budget, seed_offset=0, stall_limit=30):
            x = x0.copy()
            sigma = sigma0
            ps = np.zeros(n)
            pc = np.zeros(n)
            C = np.eye(n)
            
            # Pre-compute initial C_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            best_x = x0.copy()
            best_f = func(x0)
            evals = 1
            local_rng = np.random.default_rng(42 + seed_offset)

            last_improvement_eval = 1

            for gen in range(remaining_budget // lam + 1):
                if time.time() > deadline:
                    break
                if evals >= remaining_budget:
                    break

                # Generate population
                if gen % 5 == 0:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                Y = local_rng.standard_normal((lam, n))
                X = x[None, :] + sigma * (C_inv_sqrt @ Y.T).T
                X = np.clip(X, -5.0, 5.0)

                Fs = np.zeros(lam)
                for i in range(lam):
                    if time.time() > deadline:
                        break
                    if evals >= remaining_budget:
                        break
                    Fs[i] = func(X[i])
                    evals += 1
                    if Fs[i] < best_f:
                        best_f = Fs[i]
                        best_x = X[i].copy()
                        last_improvement_eval = evals

                if evals >= remaining_budget or time.time() > deadline:
                    break

                order = np.argsort(Fs)
                X_sorted = X[order]
                
                x_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)

                diff = (x_new - x) / (sigma * chi_n)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
                
                h_sigma = 1.0
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (evals + 1) / (lam * c_sigma))) < (1.4 + 2 / (n + 1)) * chi_n:
                    h_sigma = 1.0
                else:
                    h_sigma = 0.0

                pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff

                v = (X_sorted[:mu] - x[None, :]) / sigma
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(ps, ps) + (1 - h_sigma) * (1 - c_c) * c_c * C)
                C += c_mu * np.sum(weights[:, None, None] * v[:, :, None] * v[:, None, :], axis=0)
                C = 0.5 * (C + C.T)

                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                x = x_new
                x = np.clip(x, -5.0, 5.0)

                if evals - last_improvement_eval > stall_limit:
                    break

            return best_x, best_f, evals

        best_x = None
        best_f = float('inf')
        
        # Strategy: 
        # 1. Initial global search: Random start in [-4, 4], sigma=1.5, budget=250
        # 2. Local refinement: Start from best, sigma=0.5, budget=150
        # 3. Final polish: Start from best, sigma=0.1, budget=100
        
        # Run 1
        x0 = rng.uniform(-4, 4, n)
        rx, rf, used = cma_es(x0, 1.5, 250, seed_offset=0, stall_limit=40)
        if rf < best_f:
            best_f = rf
            best_x = rx
        budget_left = budget - used

        if budget_left > 50 and time.time() < deadline:
            # Run 2
            x0 = best_x + 0.3 * rng.standard_normal(n)
            x0 = np.clip(x0, -4.8, 4.8)
            rx, rf, used = cma_es(x0, 0.5, min(budget_left, 150), seed_offset=100, stall_limit=30)
            if rf < best_f:
                best_f = rf
                best_x = rx
            budget_left -= used

        if budget_left > 10 and time.time() < deadline:
            # Run 3
            x0 = best_x
            rx, rf, used = cma_es(x0, 0.1, min(budget_left, 100), seed_offset=200, stall_limit=20)
            if rf < best_f:
                best_f = rf
                best_x = rx

        return best_x, best_f
# EVOLVE-BLOCK-END
