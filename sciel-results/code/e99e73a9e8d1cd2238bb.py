"""Rewritten optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float).reshape(dim), -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Random sampling
        rng_init = np.random.default_rng(42)
        n_init = min(15, budget)
        for _ in range(n_init):
            if calls_made >= budget or time.time() > deadline:
                break
            x = rng_init.uniform(-4, 4, dim)
            safe_eval(x)

        def cma_es(x0, f0, max_calls, seed):
            nonlocal calls_made
            if max_calls <= 0:
                return x0, f0

            rng = np.random.default_rng(seed)
            n = max(4, int(4 + 3 * np.log(dim)))
            if n > max_calls // 3:
                n = max(4, max_calls // 3)

            x = x0.copy()
            sigma = 0.5
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            c_sigma = (4 + np.log(dim) / 2) / (dim + 4)
            c_c = (4 + np.log(dim / 2)) / (dim + 2)
            d_sigma = 1 + 2 * max(0, np.sqrt((n + 1) / (dim + 1)) - 1) / (dim + 3)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            nu = 2 / (dim + n - 2) - 1

            weights = np.log(n + 0.5) - np.arange(1, n + 1)
            weights /= np.sum(weights)
            mu = int(np.sum(weights > 0))
            mu_eff = 1.0 / np.sum(weights[:mu] ** 2)

            def get_C_sqrt(C):
                ev, evec = np.linalg.eigh(C)
                ev = np.maximum(ev, 1e-14)
                return evec @ np.diag(np.sqrt(ev)) @ evec.T, evec @ np.diag(1.0 / np.sqrt(ev)) @ evec.T

            C_sqrt, C_inv_sqrt = get_C_sqrt(C)

            pop = np.zeros((n, dim))
            pop_f = np.zeros(n)

            for i in range(n):
                if calls_made >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = np.clip(x + sigma * C_sqrt @ z, -5, 5)
                pop[i] = y
                pop_f[i] = safe_eval(y)

            idx = np.argsort(pop_f)
            pop = pop[idx]
            pop_f = pop_f[idx]
            x = pop[0].copy()
            f = pop_f[0]

            it = 0
            while calls_made < budget and time.time() < deadline:
                it += 1

                w = weights[:mu]
                x_new = np.sum(w[:, None] * pop[:mu], axis=0)

                z_mean = C_inv_sqrt @ (x_new - x) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_mean
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
                if h_sigma:
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ (x_new - x) / sigma

                delta_h = (1 - h_sigma) * (1 - c_c) * np.outer(p_c, p_c)
                C = (1 - c_c * (2 - c_c) - nu) * C
                for i in range(mu):
                    y_i = C_inv_sqrt @ (pop[i] - x) / sigma
                    C += nu * w[i] * (np.outer(y_i, y_i) + delta_h)

                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 10.0)

                C_sqrt, C_inv_sqrt = get_C_sqrt(C)

                for i in range(n):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = np.clip(x_new + sigma * C_sqrt @ z, -5, 5)
                    pop[i] = y
                    pop_f[i] = safe_eval(y)

                idx = np.argsort(pop_f)
                pop = pop[idx]
                pop_f = pop_f[idx]

                x = x_new
                f = pop_f[0]

                if sigma < 1e-10:
                    break

            return x, f

        # Phase 2: CMA-ES with restarts
        n_restarts = 4
        calls_at_start = calls_made
        for r in range(n_restarts):
            if calls_made >= budget or time.time() > deadline:
                break
            remaining = budget - calls_made
            if remaining < 15:
                break

            if r == 0:
                x0 = best_x.copy()
                f0 = best_f
            else:
                rng_r = np.random.default_rng(1000 + 100 * r)
                x0 = np.clip(best_x + 1.5 * rng_r.standard_normal(dim), -5, 5)
                f0 = safe_eval(x0)

            budget_this = min(remaining, max(20, remaining // (n_restarts - r)))
            cma_es(x0, f0, budget_this, seed=5000 + 1000 * r)

        return best_x, best_f
# EVOLVE-BLOCK-END
