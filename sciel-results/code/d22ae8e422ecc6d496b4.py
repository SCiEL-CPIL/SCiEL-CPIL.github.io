"""Optimized solver for BBOB f9 (rotated Rosenbrock) using CMA-ES + L-BFGS-B polish."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES global search + L-BFGS-B high-precision polish for rotated Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315.0  # safety margin before 330s limit
        dim = self.dim
        budget = self.budget

        # State tracking
        call_count = [0]
        best_f = [np.inf]
        best_x = [None]

        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return best_f[0]
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x_clipped.copy()
            return f

        def remaining_time():
            return deadline - time.time()

        def has_budget():
            return call_count[0] < budget and remaining_time() > 2.0

        bounds = [(-5.0, 5.0)] * dim
        rng = np.random.default_rng(42)

        overall_best_f = np.inf
        overall_best_x = np.zeros(dim)

        # Phase 1: Proper CMA-ES global search
        if has_budget():
            try:
                popsize = 4 + int(3 * np.log(dim))  # ≈ 10 for dim=5
                popsize = min(popsize, 12)
                sigma0 = 2.5
                max_evals_cma = 150  # Keep short for AOCC early progress

                x_mean = np.zeros(dim)
                sigma = sigma0
                C = np.eye(dim)

                # Strategy parameters
                mu = popsize // 2
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w /= w.sum()
                mueff = 1.0 / np.sum(w ** 2)
                c_sigma = (2 + 2.0 / (dim + 1)) / (dim + 4 + 2.0 / (dim + 1))
                c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
                mu_sigma = mueff / 2.0
                chiN = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                d = 1.0

                cma_best_f = np.inf
                cma_best_x = np.zeros(dim)
                evals_so_far = 0
                all_candidates = []  # Store top candidates for multi-start polish

                while evals_so_far < max_evals_cma and has_budget():
                    if remaining_time() < 15.0:
                        break

                    # Compute C_sqrt
                    C = (C + C.T) / 2.0
                    C += 1e-12 * np.eye(dim)
                    try:
                        C_sqrt = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-10)
                        C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))

                    # Generate population
                    population = np.zeros((popsize, dim))
                    fitness = np.zeros(popsize)
                    for i in range(popsize):
                        if not has_budget():
                            break
                        z = rng.standard_normal(dim)
                        xi = x_mean + sigma * (C_sqrt @ z)
                        population[i] = xi
                        f = counted_func(xi)
                        fitness[i] = f
                        evals_so_far += 1
                        if f < cma_best_f:
                            cma_best_f = f
                            cma_best_x = xi.copy()
                        all_candidates.append((f, xi.copy()))

                    if evals_so_far < popsize:
                        break

                    # Sort
                    order = np.argsort(fitness)
                    y = population[order]

                    # Update mean
                    x_new = np.sum(w[:, None] * y[:mu], axis=0)

                    # Update sigma
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_sqrt.T @ (x_new - x_mean) / sigma)
                    p_norm = np.linalg.norm(p_sigma)
                    if p_norm > (1.4 + 1.0 / dim) * chiN:
                        sigma *= np.exp(1.0 / (5.0 * c_sigma))
                    elif p_norm < (0.55 - 1.0 / dim) * chiN:
                        sigma *= np.exp(-1.0 / (5.0 * c_sigma))

                    # Update evolution path
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (x_new - x_mean) / sigma

                    # Update covariance
                    diff = (x_new - x_mean) / sigma
                    C = (1 + 2 * mu_sigma * c_c * (1 - c_c)) * C \
                        + (1 + 2 * mu_sigma * c_c * (1 - c_c) - c_c) * np.outer(p_c, p_c) \
                        + mu_sigma * np.outer(diff, diff)

                    # Rank-mu
                    for i in range(mu):
                        yi = (y[i] - x_mean) / sigma
                        C += 2 * mu_sigma * w[i] * np.outer(yi, yi)

                    C = (C + C.T) / 2.0
                    x_mean = x_new

                    # Check convergence
                    if sigma < 1e-12:
                        break

                if cma_best_f < overall_best_f:
                    overall_best_f = cma_best_f
                    overall_best_x = cma_best_x.copy()

            except Exception:
                if best_x[0] is not None:
                    overall_best_f = best_f[0]
                    overall_best_x = best_x[0].copy()

        # Update from global tracker
        if best_x[0] is not None and best_f[0] < overall_best_f:
            overall_best_f = best_f[0]
            overall_best_x = best_x[0].copy()

        # Phase 2: Multi-start L-BFGS-B polish from top candidates
        if has_budget() and remaining_time() > 5.0:
            # Select top 3 unique candidates for multi-start
            candidates = []
            seen = []
            all_candidates.sort(key=lambda x: x[0])
            for f_val, x_val in all_candidates:
                if len(candidates) >= 3:
                    break
                is_unique = True
                for sx in seen:
                    if np.linalg.norm(x_val - sx) < 0.1:
                        is_unique = False
                        break
                if is_unique:
                    candidates.append((f_val, x_val))
                    seen.append(x_val)

            if not candidates:
                candidates = [(overall_best_f, overall_best_x)]

            for f_val, x_val in candidates:
                if not has_budget() or remaining_time() < 3.0:
                    break
                try:
                    res = minimize(
                        counted_func,
                        np.clip(x_val, -5.0, 5.0),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
                    )
                    if res.fun < overall_best_f:
                        overall_best_f = res.fun
                        overall_best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        # Phase 3: Final tight polish
        if has_budget() and remaining_time() > 2.0:
            try:
                res = minimize(
                    counted_func,
                    overall_best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 50}
                )
                if res.fun < overall_best_f:
                    overall_best_f = res.fun
                    overall_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Final: use tracked best if better
        if best_x[0] is not None and best_f[0] < overall_best_f:
            overall_best_f = best_f[0]
            overall_best_x = best_x[0]

        if overall_best_x is None:
            if best_x[0] is not None:
                return best_x[0], best_f[0]
            return np.zeros(dim), 0.0

        return overall_best_x, overall_best_f
