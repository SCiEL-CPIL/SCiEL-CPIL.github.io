"""Bayesian Optimization with Gaussian Process surrogate for BBOB functions."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')


class Optimizer:
    """Bayesian Optimization with GP surrogate and EI acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or not time_ok():
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                fx = best_f
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial exploration with diverse points
        n_initial = 15
        X = []
        Y = []

        # Add structured initial points
        init_points = [np.zeros(n)]
        # Add random points
        for _ in range(n_initial - 1):
            init_points.append(rng.uniform(-4, 4, n))

        for x in init_points:
            if calls >= budget or not time_ok():
                break
            fx = evaluate(x)
            if fx is not None:
                X.append(x.copy())
                Y.append(fx)

        if len(X) == 0:
            x = rng.uniform(-5, 5, n)
            evaluate(x)
            return best_x, best_f

        X = np.array(X)
        Y = np.array(Y)

        # GP kernel parameters
        length_scales = np.ones(n) * 1.0
        sigma_f = 1.0
        sigma_n = 1e-8

        def compute_k(X1, X2, length_scales, sigma_f, sigma_n):
            """Compute GP kernel matrix."""
            # Normalize inputs to [0, 1] for better conditioning
            X1_norm = (X1 + 5.0) / 10.0
            X2_norm = (X2 + 5.0) / 10.0

            # Squared exponential kernel
            sq_dist = np.sum(X1_norm ** 2, axis=1)[:, None] - 2 * X1_norm @ X2_norm.T + np.sum(X2_norm ** 2, axis=1)[None, :]
            sq_dist = np.maximum(sq_dist, 0)
            sq_dist /= np.outer(length_scales, length_scales)
            K = sigma_f ** 2 * np.exp(-0.5 * sq_dist)
            if X1 is X2:
                K += sigma_n * np.eye(len(X1))
            return K

        def gp_predict(X_new, X_train, Y_train, length_scales, sigma_f, sigma_n):
            """GP mean and variance prediction."""
            K = compute_k(X_train, X_train, length_scales, sigma_f, sigma_n)
            K_s = compute_k(X_train, X_new, length_scales, sigma_f, sigma_n)

            try:
                L = np.linalg.cholesky(K + 1e-10 * np.eye(len(X_train)))
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(K + 1e-6 * np.eye(len(X_train)))

            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            v = np.linalg.solve(L, K_s.T)

            mu = v.T @ alpha
            var = sigma_f ** 2 - np.sum(v ** 2, axis=0)
            var = np.maximum(var, 1e-12)

            return mu, var

        def expected_improvement(mu, var, best_f_val):
            """Compute Expected Improvement."""
            var = np.maximum(var, 1e-12)
            sigma = np.sqrt(var)
            z = (best_f_val - mu) / sigma
            ei = (best_f_val - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        # Phase 2: Bayesian Optimization loop
        # Use remaining budget for BO iterations
        bo_budget = budget - calls
        n_bo_iter = min(bo_budget, 200)

        for bo_i in range(n_bo_iter):
            if calls >= budget or not time_ok():
                break

            # Adaptively adjust kernel parameters based on data spread
            if len(X) > 5:
                # Estimate length scales from data
                data_range = np.ptp(X, axis=0)
                data_range = np.maximum(data_range, 0.1)
                length_scales = data_range / 2.0
                length_scales = np.clip(length_scales, 0.1, 5.0)

                # Estimate signal variance
                sigma_f = np.std(Y) + 1e-6

            # Generate candidate points for EI maximization
            n_candidates = 500
            candidates = rng.uniform(-5, 5, (n_candidates, n))

            # Also add some points near current best
            if best_x is not None:
                n_local = 100
                local_candidates = best_x + rng.normal(0, 0.5, (n_local, n))
                local_candidates = np.clip(local_candidates, -5, 5)
                candidates = np.vstack([candidates, local_candidates])

            # Compute EI for all candidates
            mu, var = gp_predict(candidates, np.array(X), np.array(Y), length_scales, sigma_f, sigma_n)
            ei = expected_improvement(mu, var, best_f)

            # Select best candidate
            best_idx = np.argmax(ei)
            x_next = candidates[best_idx]

            # Evaluate
            fx = evaluate(x_next)
            if fx is None:
                break

            X.append(x_next.copy())
            Y.append(fx)

            # Periodically refine with local search
            if (bo_i + 1) % 30 == 0 and calls < budget and time_ok():
                # Try a few local steps from best point
                for _ in range(min(5, budget - calls)):
                    if not time_ok():
                        break
                    x_try = best_x + rng.normal(0, 0.1, n)
                    x_try = np.clip(x_try, -5, 5)
                    fx_try = evaluate(x_try)
                    if fx_try is None:
                        break

        # Phase 3: Final polish with coordinate descent
        if best_x is not None and calls < budget and time_ok():
            x = best_x.copy()
            f = best_f
            step = 0.3
            for cd_iter in range(min(50, budget - calls)):
                if calls >= budget or not time_ok():
                    break
                improved = False
                for j in range(n):
                    if calls >= budget or not time_ok():
                        break
                    xp = x.copy()
                    xp[j] = min(xp[j] + step, 5.0)
                    fp = evaluate(xp)
                    if fp is None:
                        break
                    if fp < f:
                        x, f = xp, fp
                        improved = True
                    else:
                        xn = x.copy()
                        xn[j] = max(xn[j] - step, -5.0)
                        fn = evaluate(xn)
                        if fn is None:
                            break
                        if fn < f:
                            x, f = xn, fn
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break

        # Phase 4: (1+1)-ES final polish
        if best_x is not None and calls < budget and time_ok():
            x = best_x.copy()
            f = best_f
            sigma = 0.1
            min_sigma = 1e-8
            for _ in range(budget - calls):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = evaluate(y)
                if fy is None:
                    break
                if fy < f:
                    x, f = y, fy
                    sigma = min(sigma * 1.1, 1.0)
                else:
                    sigma = max(sigma * 0.85, min_sigma)

        if best_x is None:
            best_x = np.zeros(n)
        return best_x, best_f
# EVOLVE-BLOCK-END
