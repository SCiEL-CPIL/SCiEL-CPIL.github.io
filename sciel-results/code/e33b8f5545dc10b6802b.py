"""Bayesian Optimization surrogate for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """Gaussian Process Bayesian Optimizer with LHS initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s kill
        rng = np.random.default_rng(42)

        # --- Phase 1: Latin Hypercube Sampling for global coverage ---
        n_lhs = min(60, self.budget // 4)
        X = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(n_lhs)
            X[:, d] = (perm + 0.5) / n_lhs * 10.0 - 5.0

        # Evaluate LHS points
        X_obs = []
        y_obs = []
        calls = 0
        for i in range(n_lhs):
            xi = X[i]
            fi = func(xi)
            X_obs.append(xi)
            y_obs.append(fi)
            calls += 1
            if time.time() > deadline:
                break

        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)

        best_idx = np.argmin(y_obs)
        x_best = X_obs[best_idx].copy()
        f_best = y_obs[best_idx]

        # Normalize y for GP
        y_min = y_obs.min()
        y_max = y_obs.max()
        if y_max - y_min < 1e-10:
            y_max = y_min + 1e-10
        y_norm = (y_obs - y_min) / (y_max - y_min)

        # --- Phase 2: GP-based Bayesian Optimization ---
        # GP hyperparameters
        length_scale = 1.0  # initial guess
        signal_var = 1.0
        noise_var = 1e-4

        def gp_predict(X_new, X_train, y_train, ls, sv, nv):
            """Predict mean and variance of GP at X_new."""
            n_train = X_train.shape[0]
            n_pred = X_new.shape[0]

            # Compute distances
            diff = X_new[:, np.newaxis, :] - X_train[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)

            # Kernel: Matern 3/2
            K = np.sqrt(5.0 / dist_sq) * np.exp(-np.sqrt(dist_sq) / ls) * sv
            # Add noise to training points
            K_train = K[:n_train, :n_train].copy()
            np.fill_diagonal(K_train, K_train.diagonal() + nv)

            try:
                L = np.linalg.cholesky(K_train)
            except np.linalg.LinAlgError:
                # Fallback to covariance matrix inversion with jitter
                K_train += 1e-6 * np.eye(n_train)
                L = np.linalg.cholesky(K_train)

            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            K_s = K[n_train:, :n_train]
            mu = K_s @ alpha
            v = np.linalg.solve(L, K_s.T)
            var = sv - np.sum(v ** 2, axis=1)
            var = np.maximum(var, 1e-10)
            return mu, var

        def expected_improvement(mu, var, f_best_norm, xi=0.01):
            """Compute EI acquisition function."""
            sigma = np.sqrt(var)
            improved = f_best_norm - mu
            z = improved / (sigma + 1e-10)
            ei = improved * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        # Adaptive restart logic
        stagnation_count = 0
        no_improve_threshold = 30

        while calls < self.budget:
            if time.time() > deadline:
                break

            # Generate candidate points
            n_candidates = 2000
            X_cand = rng.uniform(-5, 5, (n_candidates, self.dim))

            # Also add some local candidates around best point
            n_local = 500
            sigma_local = 0.5
            X_local = x_best + sigma_local * rng.standard_normal((n_local, self.dim))
            X_local = np.clip(X_local, -5, 5)
            X_cand = np.vstack([X_cand, X_local])

            # Predict GP
            try:
                mu, var = gp_predict(X_cand, X_obs, y_norm, length_scale, signal_var, noise_var)
            except Exception:
                # If GP fails, fall back to random search
                x_new = rng.uniform(-5, 5, self.dim)
                f_new = func(x_new)
                calls += 1
                X_obs = np.vstack([X_obs, x_new])
                y_obs = np.vstack([y_obs, f_new])
                y_min = y_obs.min()
                y_max = y_obs.max()
                if y_max - y_min < 1e-10:
                    y_max = y_min + 1e-10
                y_norm = (y_obs - y_min) / (y_max - y_min)
                if f_new < f_best:
                    f_best = f_new
                    x_best = x_new.copy()
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                continue

            # Compute EI
            f_best_norm = (f_best - y_min) / (y_max - y_min)
            ei = expected_improvement(mu, var, f_best_norm)

            # Select best candidate
            idx = np.argmax(ei)
            x_new = X_cand[idx]
            f_new = func(x_new)
            calls += 1

            # Update observations
            X_obs = np.vstack([X_obs, x_new])
            y_obs = np.vstack([y_obs, f_new])
            y_min = y_obs.min()
            y_max = y_obs.max()
            if y_max - y_min < 1e-10:
                y_max = y_min + 1e-10
            y_norm = (y_obs - y_min) / (y_max - y_min)

            # Update best
            if f_new < f_best:
                f_best = f_new
                x_best = x_new.copy()
                stagnation_count = 0
                # Adapt length scale: shrink when improving
                length_scale = max(0.1, length_scale * 0.9)
            else:
                stagnation_count += 1
                # Expand exploration when stagnating
                length_scale = min(3.0, length_scale * 1.05)

            # Periodic hyperparameter adaptation
            if calls % 50 == 0 and len(y_obs) > 20:
                # Simple heuristic: adjust signal variance based on y range
                signal_var = max(0.1, min(10.0, (y_max - y_min) ** 0.5))
                noise_var = max(1e-6, min(1e-2, (y_max - y_min) * 1e-4))

            # If severely stagnated, inject random restarts
            if stagnation_count > no_improve_threshold:
                n_restarts = min(10, self.budget - calls)
                for _ in range(n_restarts):
                    if calls >= self.budget or time.time() > deadline:
                        break
                    x_r = rng.uniform(-5, 5, self.dim)
                    f_r = func(x_r)
                    calls += 1
                    X_obs = np.vstack([X_obs, x_r])
                    y_obs = np.vstack([y_obs, f_r])
                    y_min = y_obs.min()
                    y_max = y_obs.max()
                    if y_max - y_min < 1e-10:
                        y_max = y_min + 1e-10
                    y_norm = (y_obs - y_min) / (y_max - y_min)
                    if f_r < f_best:
                        f_best = f_r
                        x_best = x_r.copy()
                        stagnation_count = 0
                length_scale = 1.5  # reset exploration scale

        return x_best, f_best
# EVOLVE-BLOCK-END
