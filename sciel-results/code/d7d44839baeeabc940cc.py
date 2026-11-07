"""Bayesian Optimization with GP surrogate and (1+1)-ES refinement for BBOB f19."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """GP-based Bayesian Optimization with ES refinement for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275  # safety margin
        rng = np.random.default_rng(42)
        n = self.dim

        best_x = np.zeros(n)
        best_f = float('inf')
        calls_made = 0

        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return None
            try:
                x = np.asarray(x, dtype=np.float64)
                x = np.clip(x, -5.0, 5.0)
                f = float(func(x))
                calls_made += 1
                if np.isfinite(f) and f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except Exception:
                return None

        # Phase 1: Initial random sampling to build GP training set
        X_samples = []
        Y_samples = []
        n_init = min(30, self.budget // 3)

        for i in range(n_init):
            if calls_made >= self.budget or time.time() > deadline - 10:
                break
            x = rng.uniform(-5, 5, n)
            f = safe_eval(x)
            if f is None:
                break
            X_samples.append(x)
            Y_samples.append(f)

        if not X_samples:
            return best_x, best_f

        X = np.array(X_samples)
        Y = np.array(Y_samples)

        # Phase 2: GP-based Bayesian Optimization
        def gp_predict(X_test, X_train, Y_train, length_scale=1.0, noise=1e-6):
            """Simple GP prediction with RBF kernel."""
            # Compute kernel matrix
            def rbf_kernel(A, B, ls):
                # A: (m, d), B: (k, d)
                A_sq = np.sum(A ** 2, axis=1, keepdims=True)
                B_sq = np.sum(B ** 2, axis=1, keepdims=True).T
                dist_sq = A_sq + B_sq - 2 * A @ B.T
                dist_sq = np.maximum(dist_sq, 0)
                return np.exp(-dist_sq / (2 * ls ** 2))

            K = rbf_kernel(X_train, X_train, length_scale)
            K += noise * np.eye(len(X_train))
            K_star = rbf_kernel(X_test, X_train, length_scale)

            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-6 * np.eye(len(X_train))
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    # Fallback: return mean prediction
                    mu = np.mean(Y_train) * np.ones(X_test.shape[0])
                    sigma = np.std(Y_train) * np.ones(X_test.shape[0])
                    return mu, sigma

            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            mu = K_star @ alpha
            v = np.linalg.solve(L, K_star.T)
            sigma_sq = rbf_kernel(X_test, X_test, length_scale).diagonal() - np.sum(v ** 2, axis=0)
            sigma = np.sqrt(np.maximum(sigma_sq, 1e-10))
            return mu, sigma

        def expected_improvement(mu, sigma, best_y, xi=0.01):
            """Expected Improvement acquisition function."""
            sigma = np.maximum(sigma, 1e-10)
            z = (best_y - mu) / sigma
            from scipy.stats import norm
            ei = (best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        def optimize_acquisition(X_train, Y_train, n_candidates=200, length_scale=1.0):
            """Find best candidate using EI acquisition."""
            best_y = np.min(Y_train)
            # Generate candidates
            candidates = rng.uniform(-5, 5, (n_candidates, n))
            mu, sigma = gp_predict(candidates, X_train, Y_train, length_scale)
            ei = expected_improvement(mu, sigma, best_y)
            best_idx = np.argmax(ei)
            return candidates[best_idx]

        # Adaptive length scale based on observed scale
        y_range = np.ptp(Y)
        if y_range < 1e-8:
            y_range = 1.0

        for bo_iter in range(50):
            if calls_made >= self.budget or time.time() > deadline - 15:
                break

            # Adaptive length scale: decrease as we converge
            progress = 1.0 - (calls_made / self.budget)
            length_scale = 0.5 + 1.5 * progress  # Start wide, narrow down

            # Get next point from GP
            x_next = optimize_acquisition(X, Y, n_candidates=150, length_scale=length_scale)
            f_next = safe_eval(x_next)
            if f_next is None:
                break

            X = np.vstack([X, x_next])
            Y = np.append(Y, f_next)

            # If we've made good progress, switch to ES refinement
            if calls_made > self.budget * 0.6:
                break

        # Phase 3: (1+1)-ES refinement from best point
        if calls_made < self.budget and time.time() < deadline - 3:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            success_count = 0
            total_count = 0
            window_size = 15

            while calls_made < self.budget and time.time() < deadline - 1:
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = safe_eval(y)
                if fy is None:
                    break

                total_count += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.4
                    success_count += 1
                else:
                    sigma *= 0.85

                if total_count % window_size == 0:
                    rate = success_count / window_size
                    if rate > 0.25:
                        sigma *= 1.1
                    elif rate < 0.08:
                        sigma *= 0.9
                    success_count = 0

                if sigma < 1e-12:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
