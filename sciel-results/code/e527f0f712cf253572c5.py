"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with time guard for BBOB f9 (Rosenbrock rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def check_deadline():
            return time.time() > deadline

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(42)

        # Phase 1: Quick random sampling
        n_init = min(10, budget)
        for i in range(n_init):
            if check_deadline():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_used += 1
            update_best(x, fx)

        # Phase 2: CMA-ES with restarts
        def run_cmaes(x0, sigma0, max_calls):
            nonlocal calls_used, best_x, best_f
            n = dim
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            x = x0.copy()
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            calls_local = 0
            gen = 0
            chi_n = np.sqrt(n) * (1 + 1 / (4 * n) - 1 / (21 * n ** 2))

            while calls_local < max_calls and not check_deadline():
                # Generate population
                ys = []
                fys = []
                for i in range(lam):
                    if calls_local >= max_calls or check_deadline():
                        break
                    z = rng.standard_normal(n)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_local += 1
                    calls_used += 1
                    update_best(y, fy)
                    ys.append(y)
                    fys.append(fy)

                if len(ys) < mu:
                    break

                # Select mu best
                idx = np.argsort(fys)[:mu]
                ys_mu = np.array([ys[i] for i in idx])

                # Weighted mean
                m = np.sum(w[:, None] * ys_mu, axis=0)

                # Update evolution paths
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ (D * ((m - x) / sigma)))
                if np.linalg.norm(p_sigma) / chi_n < (1.4 + 2 / (n + 1)) * np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))):
                    p_c = (1 - c_c) * p_c + c_c * (m - x) / sigma

                # Update covariance via B and D
                # C = B D^2 B^T
                C = B @ np.diag(D ** 2) @ B.T

                # Standard CMA-ES update
                delta_hs = 1 - (1 - c_c) ** (gen + 1)
                C_new = (1 - c_1 - c_mu) * C \
                        + c_1 * (p_c @ p_c + delta_hs * np.eye(n)) \
                        + c_mu / mu_eff * np.sum(w[:, None, None] * np.outer(ys_mu - m, ys_mu - m), axis=0)

                # Ensure positive definiteness
                try:
                    eigvals, eigvecs = np.linalg.eigh(C_new)
                    eigvals = np.maximum(eigvals, 1e-10)
                    C_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C_new = C

                B, D = eigvecs, np.sqrt(eigvals)

                # Update x and sigma
                x = m
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = max(1e-10, min(sigma, 50.0))
                gen += 1

            return x, calls_local

        # CMA-ES restarts
        n_restarts = 3
        remaining = budget - calls_used
        calls_per_restart = max(10, remaining // n_restarts)

        for r in range(n_restarts):
            if check_deadline() or calls_used >= budget:
                break

            if r == 0:
                x0 = best_x.copy()
                sigma0 = 1.0
            else:
                x0 = best_x + rng.normal(0, 1.5 + r, dim)
                x0 = np.clip(x0, -5, 5)
                sigma0 = 1.5 + r

            run_cmaes(x0, sigma0, calls_per_restart)

        return best_x, best_f
# EVOLVE-BLOCK-END
