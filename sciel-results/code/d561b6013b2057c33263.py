"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Surrogate-assisted optimizer with CMA-ES fallback for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        call_count = [0]
        all_x = []
        all_f = []
        best_x = None
        best_f = np.inf

        def wrapped(x):
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            call_count[0] += 1
            fx = float(func(x))
            all_x.append(x.copy())
            all_f.append(fx)
            return fx

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(123)

        # Phase 1: Initial sampling
        n_init = min(20, budget // 10)
        for i in range(n_init):
            if time.time() > deadline - 20.0:
                break
            if call_count[0] >= budget - 10:
                break
            x = rng.uniform(-5, 5, dim)
            fx = wrapped(x)
            update_best(x, fx)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)
            update_best(best_x, best_f)

        # Phase 2: Surrogate-assisted search using Expected Improvement
        # Build a simple GP-like surrogate using RBF kernel
        def rbf_kernel(X1, X2, length_scale=1.0):
            # X1: (n1, d), X2: (n2, d)
            d2 = np.sum(X1**2, axis=1)[:, None] - 2 * X1 @ X2.T + np.sum(X2**2, axis=1)[None, :]
            return np.exp(-0.5 * d2 / (length_scale ** 2))

        def gp_predict(X_new, X_train, y_train, length_scale=1.0, noise=1e-8):
            """Simple GP prediction with RBF kernel."""
            K = rbf_kernel(X_train, X_train, length_scale) + noise * np.eye(len(X_train))
            K_star = rbf_kernel(X_train, X_new, length_scale)
            try:
                alpha = np.linalg.solve(K, y_train)
                mean = K_star.T @ alpha
                K_star_star = rbf_kernel(X_new, X_new, length_scale)
                var = np.diag(K_star_star) - np.sum(K_star.T * np.linalg.solve(K, K_star.T), axis=0)
                var = np.maximum(var, 1e-10)
                return mean, np.sqrt(var)
            except:
                # Fallback: nearest neighbor
                d = np.sum((X_new[:, None, :] - X_train[None, :, :])**2, axis=2)
                idx = np.argmin(d, axis=1)
                return y_train[idx], np.zeros(len(X_new))

        def expected_improvement(mean, std, best_f, xi=0.01):
            z = (best_f - mean - xi) / std
            from scipy.stats import norm
            ei = std * (z * norm.pdf(z) + norm.cdf(z))
            return ei

        # Surrogate phase: iterate for a while
        surrogate_calls = 0
        max_surrogate_calls = min(150, budget - call_count[0] - 50)
        no_improve_surrogate = 0

        for it in range(max_surrogate_calls):
            if time.time() > deadline - 30.0:
                break
            if call_count[0] >= budget - 20:
                break

            # Build training data
            X_train = np.array(all_x)
            y_train = np.array(all_f)
            
            # Normalize y for better GP behavior
            y_mean = y_train.mean()
            y_std = y_train.std() + 1e-10
            y_norm = (y_train - y_mean) / y_std

            # Generate candidate points
            n_cands = 200
            cands = []
            # Mix of random and local perturbations
            for j in range(n_cands // 2):
                cands.append(rng.uniform(-5, 5, dim))
            for j in range(n_cands // 2):
                cands.append(np.clip(best_x + rng.normal(0, 0.5, dim), -5, 5))
            cands = np.array(cands)

            # Predict
            mean_pred, std_pred = gp_predict(cands, X_train, y_norm, length_scale=1.5)
            
            # Convert back to original scale for EI calculation
            mean_orig = mean_pred * y_std + y_mean
            std_orig = std_pred * y_std

            # EI
            ei = expected_improvement(mean_orig, std_orig, best_f, xi=0.01)
            
            # Select best candidate by EI
            idx_best = np.argmax(ei)
            x_new = cands[idx_best]
            fx = wrapped(x_new)
            update_best(x_new, fx)
            surrogate_calls += 1

            if fx < best_f + 1e-10:
                no_improve_surrogate = 0
            else:
                no_improve_surrogate += 1

            if no_improve_surrogate > 15:
                break

        # Phase 3: CMA-ES for local refinement
        def cma_es(x0, sigma, remaining_budget):
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

            for gen in range(remaining_budget // pop_size + 1):
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

        # Run CMA-ES with remaining budget
        remaining = budget - call_count[0]
        if remaining > 2 * dim and time.time() < deadline - 20.0:
            cma_es(best_x.copy(), 1.0, remaining)

        # Phase 4: Final Nelder-Mead polish
        if time.time() < deadline - 10.0 and call_count[0] < budget - 5:
            try:
                result = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'fatol': 1e-10, 'xatol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except:
                pass

        # Phase 5: If budget remains, another CMA-ES with smaller sigma
        remaining = budget - call_count[0]
        if remaining > 2 * dim and time.time() < deadline - 10.0:
            cma_es(best_x.copy(), 0.3, remaining)

        # Phase 6: Final polish
        if time.time() < deadline - 5.0 and call_count[0] < budget - 3:
            try:
                result = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'fatol': 1e-12, 'xatol': 1e-10}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
