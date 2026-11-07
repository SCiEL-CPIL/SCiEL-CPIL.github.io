"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: CMA-ES for global search + Nelder-Mead for local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        # Wrap func to count calls and clip
        call_count = [0]

        def wrapped(x):
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            call_count[0] += 1
            return float(func(x))

        best_x = None
        best_f = np.inf

        def cma_es(x0, sigma, remaining_budget):
            nonlocal best_x, best_f, call_count
            n = dim
            # CMA-ES parameters
            mu = max(1, n // 2)  # number of selected individuals
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / ((n + 1.3) ** 2 + mueff)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # State
            mean = np.array(x0, dtype=np.float64)
            sigma = float(sigma)
            ps = np.zeros(n)
            C = np.eye(n)

            pop_size = 2 * mu
            no_improve = 0

            for gen in range(remaining_budget // pop_size + 1):
                if time.time() > deadline - 10.0:
                    break
                if call_count[0] >= budget - 5:
                    break

                # Generate population using eigendecomposition
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    D_sqrt = np.sqrt(eigvals)
                    Y = rng.standard_normal((pop_size, n))
                    X = mean[None, :] + sigma * (Y * D_sqrt[None, :]) @ eigvecs.T
                except:
                    Y = rng.standard_normal((pop_size, n))
                    X = mean[None, :] + sigma * Y

                X = np.clip(X, -5.0, 5.0)

                # Evaluate
                F = np.array([wrapped(xi) for xi in X])

                # Sort
                idx = np.argsort(F)
                X_selected = X[idx[:mu]]
                F_selected = F[idx[:mu]]

                # Update best
                if F_selected[0] < best_f:
                    best_f = F_selected[0]
                    best_x = X_selected[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Update mean
                new_mean = np.sum(weights[:, None] * X_selected, axis=0)
                new_mean = np.clip(new_mean, -5.0, 5.0)

                # Update evolution path
                diff_mean = (new_mean - mean) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * diff_mean

                # Update covariance matrix
                delta_d = (1 + c_sigma * (2 - c_sigma)) * (np.linalg.norm(ps) ** 2 / n - 1)
                rank_one = np.outer(ps, ps)
                rank_mu = np.sum(weights[:, None, None] * X_selected[:, :, None] * X_selected[:, None, :], axis=0)
                C = (1 + c_1 * delta_d - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2.0
                try:
                    eigvals_c, _ = np.linalg.eigh(C)
                    if np.any(eigvals_c < 1e-20):
                        C += 1e-10 * np.eye(n)
                except:
                    pass

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                mean = new_mean

                # Check for stagnation
                if no_improve > 30:
                    break

            return best_x, best_f

        rng = np.random.default_rng(42)

        # Phase 1: Initial random evaluations
        n_init = min(15, budget // 10)
        for i in range(n_init):
            if time.time() > deadline - 15.0:
                break
            if call_count[0] >= budget - 10:
                break
            x = rng.uniform(-5, 5, dim)
            fx = wrapped(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)

        # Phase 2: CMA-ES for global search
        remaining = budget - call_count[0]
        if remaining > 2 * dim and time.time() < deadline - 20.0:
            x0 = best_x.copy()
            sigma0 = 1.5
            cma_es(x0, sigma0, remaining)

        # Phase 3: Nelder-Mead local polish
        if time.time() < deadline - 15.0 and call_count[0] < budget - 10:
            try:
                result = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'fatol': 1e-10, 'xatol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except:
                pass

        # Phase 4: If budget remains, another CMA-ES restart with smaller sigma
        remaining = budget - call_count[0]
        if remaining > 2 * dim and time.time() < deadline - 10.0:
            x0 = best_x.copy()
            sigma0 = 0.3
            cma_es(x0, sigma0, remaining)

        # Phase 5: Final Nelder-Mead polish if budget allows
        if time.time() < deadline - 5.0 and call_count[0] < budget - 5:
            try:
                result = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'fatol': 1e-12, 'xatol': 1e-10}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
