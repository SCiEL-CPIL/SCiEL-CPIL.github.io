"""Surrogate-assisted optimizer with CMA-ES for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Surrogate-guided optimizer with CMA-ES and local polish for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        call_count = [0]
        X_obs = []  # list of (x, f) pairs
        best_x = None
        best_f = np.inf

        def wrapped(x):
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            call_count[0] += 1
            fx = float(func(x))
            X_obs.append((x.copy(), fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Surrogate model: RBF kernel Gaussian Process ---
        def rbf_kernel(X1, X2, length_scale=1.0):
            # X1: (n1, d), X2: (n2, d)
            sq_dists = np.sum(X1 ** 2, axis=1)[:, None] + np.sum(X2 ** 2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0.0)
            return np.exp(-sq_dists / (2 * length_scale ** 2))

        def gp_predict(X_new, X_train, y_train, length_scale=1.0, noise=1e-8):
            """Return mean and variance of GP prediction at X_new."""
            n = len(X_train)
            if n == 0:
                return np.zeros(len(X_new)), np.ones(len(X_new))
            K = rbf_kernel(X_train, X_train, length_scale) + noise * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            except np.linalg.LinAlgError:
                K_reg = K + 1e-6 * np.eye(n)
                try:
                    L = np.linalg.cholesky(K_reg)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                except:
                    return np.mean(y_train), np.var(y_train)

            K_s = rbf_kernel(X_new, X_train, length_scale)
            mu = K_s @ alpha
            v = np.linalg.solve(L, K_s.T)
            K_ss = rbf_kernel(X_new, X_new, length_scale)
            var = np.diag(K_ss) - np.sum(v ** 2, axis=1)
            var = np.maximum(var, 1e-10)
            return mu, var

        def expected_improvement(mu, var, best_y, xi=0.01):
            """EI acquisition for minimization."""
            sigma = np.sqrt(var)
            improved = best_y - mu - xi
            with np.errstate(divide='ignore', invalid='ignore'):
                z = improved / (sigma + 1e-15)
                ei = improved * np.where(sigma > 1e-12, 1.0, 0.0) + sigma * np.where(sigma > 1e-12, z * 0.3989422804014327 + 1.0, 0.0)
                ei = np.where(sigma > 1e-12, ei, 0.0)
            return ei

        def surrogate_select(n_candidates, length_scale=1.0):
            """Select best candidate using EI acquisition."""
            if len(X_obs) < 2:
                # Not enough data, return random points
                pts = rng.uniform(-5, 5, (n_candidates, dim))
                return pts

            X_train = np.array([p[0] for p in X_obs])
            y_train = np.array([p[1] for p in X_obs])
            best_y = np.min(y_train)

            # Generate candidates: mix of random, around best, and grid
            n_rand = max(1, n_candidates // 3)
            n_around = max(1, n_candidates // 3)
            n_grid = n_candidates - n_rand - n_around

            candidates = []
            # Random
            candidates.append(rng.uniform(-5, 5, (n_rand, dim)))
            # Around best point
            if best_x is not None:
                sigma_local = max(0.1, 0.5 * (1 - call_count[0] / budget))
                candidates.append(np.clip(best_x[None, :] + sigma_local * rng.standard_normal((n_around, dim)), -5, 5))
            # Simple grid around best
            if n_grid > 0 and best_x is not None:
                grid_pts = []
                step = 1.0
                for i in range(n_grid):
                    pt = best_x.copy()
                    axis = i % dim
                    direction = 1 if (i // dim) % 2 == 0 else -1
                    pt[axis] += direction * step * (i // dim + 1)
                    pt = np.clip(pt, -5, 5)
                    grid_pts.append(pt)
                candidates.append(np.array(grid_pts))

            all_cands = np.vstack(candidates)
            all_cands = np.clip(all_cands, -5, 5)

            mu, var = gp_predict(all_cands, X_train, y_train, length_scale)
            ei = expected_improvement(mu, var, best_y)

            # Pick top candidates by EI
            idx = np.argsort(-ei)[:n_candidates]
            return all_cands[idx]

        rng = np.random.default_rng(12345)

        # Phase 1: Initial diverse sampling
        n_init = min(20, budget // 10)
        for i in range(n_init):
            if time.time() > deadline - 20.0 or call_count[0] >= budget - 10:
                break
            if i < dim:
                # Sample along axes
                x = np.zeros(dim)
                x[i] = rng.choice([-4, -2, 0, 2, 4])
            else:
                x = rng.uniform(-5, 5, dim)
            wrapped(x)

        if best_x is None:
            best_x = np.zeros(dim)
            wrapped(best_x)

        # Phase 2: Surrogate-guided search
        # Use surrogate for early-to-mid phase, then switch to CMA-ES
        surrogate_end = max(50, budget // 3)
        while call_count[0] < surrogate_end and time.time() < deadline - 30.0:
            # Adapt length scale based on spread of observed points
            if len(X_obs) >= 2:
                X_arr = np.array([p[0] for p in X_obs])
                spread = np.mean(np.std(X_arr, axis=0))
                ls = max(0.3, min(2.0, spread * 0.5))
            else:
                ls = 1.0

            n_cand = 10
            cands = surrogate_select(n_cand, ls)
            # Evaluate the best EI candidate
            if len(cands) > 0:
                x_next = cands[0]
                wrapped(x_next)

            # Occasionally evaluate a second candidate
            if len(cands) > 1 and call_count[0] < surrogate_end - 5:
                x_next2 = cands[1]
                wrapped(x_next2)

        # Phase 3: CMA-ES for global refinement
        remaining = budget - call_count[0]
        if remaining > 2 * dim and time.time() < deadline - 25.0:
            def cma_es(x0, sigma, max_evals):
                nonlocal best_x, best_f, call_count
                n = dim
                mu = max(1, n // 2)
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mueff = 1.0 / np.sum(weights ** 2)
                c_sigma = (mueff + 2) / ((n + 1.3) ** 2 + mueff)
                d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
                c_1 = 2 / ((n + 1.3) ** 2 + mueff)
                c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
                chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

                mean = np.array(x0, dtype=np.float64)
                sigma = float(sigma)
                ps = np.zeros(n)
                C = np.eye(n)

                pop_size = 2 * mu
                no_improve = 0

                for gen in range(max_evals // pop_size + 1):
                    if time.time() > deadline - 15.0:
                        break
                    if call_count[0] >= budget - 5:
                        break

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

                    F = np.array([wrapped(xi) for xi in X])

                    idx = np.argsort(F)
                    X_selected = X[idx[:mu]]
                    F_selected = F[idx[:mu]]

                    if F_selected[0] < best_f:
                        best_f = F_selected[0]
                        best_x = X_selected[0].copy()
                        no_improve = 0
                    else:
                        no_improve += 1

                    new_mean = np.sum(weights[:, None] * X_selected, axis=0)
                    new_mean = np.clip(new_mean, -5.0, 5.0)

                    diff_mean = (new_mean - mean) / sigma
                    ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * diff_mean

                    delta_d = (1 + c_sigma * (2 - c_sigma)) * (np.linalg.norm(ps) ** 2 / n - 1)
                    rank_one = np.outer(ps, ps)
                    rank_mu = np.sum(weights[:, None, None] * X_selected[:, :, None] * X_selected[:, None, :], axis=0)
                    C = (1 + c_1 * delta_d - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

                    C = (C + C.T) / 2.0
                    try:
                        eigvals_c, _ = np.linalg.eigh(C)
                        if np.any(eigvals_c < 1e-20):
                            C += 1e-10 * np.eye(n)
                    except:
                        pass

                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                    mean = new_mean

                    if no_improve > 25:
                        break

                return best_x, best_f

            cma_es(best_x.copy(), 1.0, remaining)

        # Phase 4: Local polish with Nelder-Mead
        if time.time() < deadline - 10.0 and call_count[0] < budget - 10:
            try:
                result = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'fatol': 1e-12, 'xatol': 1e-10}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except:
                pass

        # Phase 5: Final small CMA-ES if budget remains
        remaining = budget - call_count[0]
        if remaining > 2 * dim and time.time() < deadline - 5.0:
            cma_es(best_x.copy(), 0.2, remaining)

        # Phase 6: Final polish
        if time.time() < deadline - 2.0 and call_count[0] < budget - 3:
            try:
                result = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'fatol': 1e-14, 'xatol': 1e-12}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
