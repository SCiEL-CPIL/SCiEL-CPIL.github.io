"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive restart CMA-ES with population for Rastrigin-like landscape."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng, time_limit, t_start):
        """Run CMA-ES for a given budget, return best_x, best_f."""
        dim = self.dim
        # Population size
        lam = 4 + int(3 * np.log(dim))
        # Weights
        w = np.log(lam / 2 + 1) - np.log(np.arange(1, lam + 1))
        w = w / w.sum()
        mu = int(np.sum(w >= w[-1]))
        wmu = w[:mu] / w[:mu].sum()
        mu_eff = 1.0 / np.sum(wmu ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (2 - c_c + c_sigma)
        c_mu = (2 / ((dim + 2) ** 2 + mu_eff)) * (c_c + c_sigma - c_1) if mu_eff > 0 else 0.0

        x = x0.copy()
        sigma = sigma0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        H = np.zeros((dim, dim))
        H = np.eye(dim) / np.sqrt(dim)

        best_x = x.copy()
        best_f = func(x)
        calls = 1

        for gen in range(budget // lam + 1):
            # Check time
            if time.time() - t_start > time_limit - 2.0:
                break

            # Generate population
            offspring = []
            fvals = []
            for i in range(lam):
                if calls + 1 > self.budget:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (H @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                offspring.append(y)
                fvals.append(fy)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(offspring) == 0:
                break

            # Sort
            order = np.argsort(fvals)
            offspring = [offspring[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Update mean
            x_old = x.copy()
            for i in range(mu):
                x += wmu[i] * (offspring[i] - x_old)

            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (H.T @ (x - x_old) / sigma)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update C
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x - x_old) / sigma
            delta = 1 + 2 * max(0, (1 - c_c) ** 0.5 - 1) + c_c
            rank_c = np.outer(p_c, p_c) / dim
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (offspring[i] - x_old) / sigma
                rank_mu += wmu[i] * np.outer(y_i, y_i)
            C = (1 + c_1 * (dim + 2 - c_mu) - c_mu) * C + c_1 * (dim + 2 - c_mu) * (np.eye(dim) - C) + c_mu * rank_mu
            # Simplified: standard CMA-ES update
            C = (1 - c_1 - c_mu) * C + c_1 * (np.eye(dim) - C + rank_c) + c_mu * rank_mu

            # Ensure symmetry and positive definite-ish
            C = (C + C.T) / 2

            # Update H
            H = np.linalg.cholesky(C + 1e-12 * np.eye(dim)) if dim <= 20 else np.eye(dim)
            # For dim=5, Cholesky is fine
            try:
                H = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except:
                H = np.eye(dim)

        return best_x, best_f, calls

    def __call__(self, func):
        t_start = time.time()
        time_limit = 280.0  # Safety margin
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Multi-start with adaptive restart CMA-ES
        # First do a quick random search to find a good starting point
        best_x = None
        best_f = np.inf
        calls_used = 0

        # Phase 1: Random sampling to find a decent starting point
        n_random = min(50, budget // 10)
        for i in range(n_random):
            x = rng.uniform(-4, 4, dim)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Adaptive restart CMA-ES
        # Start with larger sigma, decrease on restart
        sigma_start = 3.0
        remaining_budget = budget - calls_used

        # Number of restarts
        n_restarts = 5
        budget_per_restart = remaining_budget // n_restarts

        for restart in range(n_restarts):
            if calls_used >= budget:
                break
            if time.time() - t_start > time_limit - 5.0:
                break

            # Perturb the global best with decreasing noise
            noise = sigma_start * (0.5 ** restart)
            if restart == 0:
                x0 = best_x.copy()
            else:
                x0 = best_x + noise * rng.standard_normal(dim)
                x0 = np.clip(x0, -4, 4)

            sigma0 = sigma_start * (0.7 ** restart)

            # Run CMA-ES for this restart
            cur_budget = min(budget_per_restart, budget - calls_used)
            if cur_budget < 10:
                break

            x_res, f_res, calls_this = self._cma_es(
                func, x0, sigma0, cur_budget, rng, time_limit, t_start
            )
            calls_used += calls_this

            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

        # Phase 3: Fine-tuning with small local search around best
        if calls_used < budget and time.time() - t_start < time_limit - 10.0:
            # Do a few more small steps
            sigma = 0.1
            for _ in range(min(50, budget - calls_used)):
                if time.time() - t_start > time_limit - 2.0:
                    break
                x_new = best_x + sigma * rng.standard_normal(dim)
                x_new = np.clip(x_new, -5, 5)
                fx_new = func(x_new)
                calls_used += 1
                if fx_new < best_f:
                    best_f = fx_new
                    best_x = x_new.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
