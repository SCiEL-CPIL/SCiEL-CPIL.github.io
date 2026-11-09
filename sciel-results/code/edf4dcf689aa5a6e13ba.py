"""Rewritten program for BBOB f7 (Step ellipsoidal) using multi-restart CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart CMA-ES with Nelder-Mead polish for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        x_best = np.clip(rng.uniform(-5, 5, dim), -5, 5)
        f_best = func(x_best)
        calls = 1

        def cma_es(x0, local_budget, seed):
            """Run CMA-ES from x0 with given local budget."""
            nonlocal calls, f_best, x_best

            if calls >= budget or time.time() >= deadline:
                return x_best, f_best, calls

            local_rng = np.random.default_rng(seed)
            lam = max(4 + int(3 * np.log(dim)), 8)
            sigma = 2.0
            x_mean = x0.copy()

            # Covariance matrix (initially identity)
            C = np.eye(dim)
            # Step size adaptation
            c_sigma = (lam + 2) / (dim + lam + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (dim + 1)) - 1)
            cs = (dim + lam / 2 - 2) / (dim + 5 + lam / 2)
            cmu = min(2 - cs, 2 / ((dim + 2) ** 2 + lam))

            ps = np.zeros(dim)
            pc = np.zeros(dim)

            # Cholesky of C (identity initially)
            C_sqrt = np.eye(dim)

            local_calls = 0

            while local_calls < local_budget and calls < budget and time.time() < deadline:
                # Generate population
                x_pop = np.empty((lam, dim))
                for i in range(lam):
                    z = local_rng.standard_normal(dim)
                    x_pop[i] = x_mean + sigma * (C_sqrt @ z)
                    x_pop[i] = np.clip(x_pop[i], -5, 5)

                # Evaluate
                f_pop = np.empty(lam)
                for i in range(lam):
                    if calls >= budget or time.time() >= deadline:
                        break
                    f_pop[i] = func(x_pop[i])
                    calls += 1
                    local_calls += 1
                    if f_pop[i] < f_best:
                        f_best = f_pop[i]
                        x_best = x_pop[i].copy()

                if local_calls == 0:
                    break

                # Sort
                idx = np.argsort(f_pop)[:lam // 2 + 1]
                mu = lam // 2 + 1
                weights = np.log(mu + 0.5) - np.log(1 + np.arange(mu))
                weights /= weights.sum()
                mueff = 1 / np.sum(weights ** 2)

                # Update mean
                x_mean_old = x_mean.copy()
                x_mean = np.sum(weights[:, None] * x_pop[idx, :], axis=0)

                # Update ps
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_sqrt.T @ ((x_mean - x_mean_old) / sigma))

                # Update pc
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (local_calls + 1))) < (1.4 + 2 / (dim + 1)) * cs:
                    pc = (1 - cs) * pc + cs * (x_mean - x_mean_old) / sigma

                # Update C
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (local_calls + 1)) < (1.4 + 2 / (dim + 1)) * cs
                rank1 = (1 + (2 * hsig - 1) * c_sigma * (2 - c_sigma)) * np.outer(pc, pc) + (c_sigma * (2 - c_sigma) - cs * (2 - cs)) * np.outer(ps, ps)
                rankmu = np.zeros((dim, dim))
                for i in range(mu):
                    y = (x_pop[idx[i]] - x_mean_old) / sigma
                    rankmu += np.outer(weights[i] * y, weights[i] * y)
                C = (1 + cs * (cmu - 1) + 2 * cmu) * C - rank1 + cmu * rankmu

                # Cholesky decomposition
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)

                # Adapt sigma
                sigma *= np.exp(c_sigma * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1) * d_sigma)
                sigma = np.clip(sigma, 1e-6, 10.0)

            return x_best, f_best, calls

        # Phase 1: Multi-start exploration
        n_starts = 6
        per_start_budget = min(60, (budget - calls) // n_starts)
        for i in range(n_starts):
            if calls >= budget or time.time() >= deadline:
                break
            if i == 0:
                x0 = x_best.copy()
            else:
                x0 = np.clip(rng.uniform(-4, 4, dim), -5, 5)
            cma_es(x0, per_start_budget, seed=1000 + i)

        # Phase 2: Nelder-Mead polish around best
        if calls < budget and time.time() < deadline:
            x0 = x_best.copy()
            # Create initial simplex
            x0_simplex = [x0.copy()]
            for i in range(dim):
                x_i = x0.copy()
                x_i[i] += 0.5 if x0[i] < 4.5 else -0.5
                x0_simplex.append(np.clip(x_i, -5, 5))

            def nm_func(x):
                nonlocal calls
                if calls >= budget or time.time() >= deadline:
                    raise RuntimeError("budget exceeded")
                f = func(x)
                calls += 1
                if f < f_best:
                    f_best = f
                    x_best = x.copy()
                return f

            try:
                res = minimize(nm_func, x0, method='Nelder-Mead',
                               options={'maxiter': 200, 'fatol': 1e-10, 'xatol': 1e-8})
            except Exception:
                pass

        # Phase 3: Fine-tuning with small-step CMA-ES
        if calls < budget and time.time() < deadline:
            remaining = budget - calls
            if remaining > 20:
                cma_es(x_best, min(remaining, 100), seed=9999)

        return x_best, f_best
# EVOLVE-BLOCK-END
