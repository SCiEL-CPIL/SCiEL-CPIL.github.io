"""Bayesian Optimization surrogate for Büche-Rastrigin f4."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Bayesian Optimization with Gaussian Process surrogate."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290.0  # safety margin before 300s
        rng = np.random.default_rng(42)

        # Initial exploration: LHS + center
        n_init = 20
        X = np.zeros((n_init, self.dim))
        for i in range(n_init):
            X[i] = rng.uniform(-5, 5, self.dim)
        # Add center and some structured points
        X[0] = np.zeros(self.dim)
        X[1] = np.array([2.0] * self.dim)
        X[2] = np.array([-2.0] * self.dim)
        X[3] = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        X[4] = np.array([-1.0, 1.0, -1.0, 1.0, -1.0])

        Y = np.array([func(x) for x in X])

        best_val = Y.min()
        best_x = X[Y.argmin()]

        calls = n_init
        batch_size = 5

        # GP hyperparameters
        length_scales = np.ones(self.dim) * 2.0
        signal_var = 100.0
        noise_var = 1e-4

        def gp_predict(X_new, length_scales, signal_var, noise_var, Y_obs):
            """Predict mean and variance using a simple RBF kernel."""
            n = len(Y_obs)
            # Compute kernel matrix K(X_obs, X_new)
            X_obs = Y_obs  # placeholder, we'll use stored X
            # We'll implement this inline in the main loop for clarity
            pass

        # We'll implement a lightweight GP inline
        def compute_K(A, B, ls, sv):
            """RBF kernel between rows of A and B."""
            A2 = np.sum(A ** 2, axis=1, keepdims=True)
            B2 = np.sum(B ** 2, axis=1, keepdims=True)
            D2 = A2 + B2.T - 2.0 * A @ B.T
            D2 = np.maximum(D2, 0)
            # Scale by length scales
            D2_scaled = D2 / (2.0 * ls ** 2)
            return sv * np.exp(-D2_scaled)

        def gp_predict_mean_var(X_new, X_obs, Y_obs, ls, sv, nv):
            n = len(X_obs)
            K_obs_obs = compute_K(X_obs, X_obs, ls, sv) + nv * np.eye(n)
            K_new_obs = compute_K(X_new, X_obs, ls, sv)
            # Solve K_obs_obs @ alpha = Y_obs
            try:
                alpha = np.linalg.solve(K_obs_obs, Y_obs)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K_obs_obs, Y_obs, rcond=None)[0]
            mu = K_new_obs @ alpha
            # Variance
            K_new_new = compute_K(X_new, X_new, ls, sv)
            var = np.diag(K_new_new) - np.sum(K_new_obs ** 2 * alpha[:, None], axis=0)
            var = np.maximum(var, 1e-10)
            return mu, var

        # Main BO loop
        while calls < self.budget:
            if time.time() > deadline - 5.0:
                break

            # Generate candidate points for acquisition
            n_cand = 200
            X_cand = rng.uniform(-5, 5, (n_cand, self.dim))

            # Predict
            mu, var = gp_predict_mean_var(X_cand, X, Y, length_scales, signal_var, noise_var)

            # Expected Improvement acquisition
            best_y = Y.min()
            improvement = best_y - mu
            sigma = np.sqrt(var)
            sigma = np.maximum(sigma, 1e-8)
            Z = improvement / sigma
            from scipy.stats import norm
            EI = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

            # Select top batch_size candidates, ensuring diversity
            # Simple approach: pick top 1, then greedily add diverse points
            selected = []
            scores = EI.copy()
            for _ in range(min(batch_size, self.budget - calls)):
                idx = np.argmax(scores)
                selected.append(X_cand[idx])
                # Reduce scores of points close to selected
                dist = np.sum((X_cand - X_cand[idx]) ** 2, axis=1)
                scores -= 100.0 / (1.0 + dist)

            # Evaluate selected points
            for x_new in selected:
                if calls >= self.budget or time.time() > deadline - 2.0:
                    break
                f_new = func(x_new)
                calls += 1
                X = np.vstack([X, x_new])
                Y = np.append(Y, f_new)
                if f_new < best_val:
                    best_val = f_new
                    best_x = x_new

            # Adaptively adjust length scales based on best improvement region
            if len(X) > 10:
                # Roughly estimate local scale from recent improvements
                recent = X[-10:]
                diffs = np.diff(recent, axis=0)
                if len(diffs) > 0:
                    local_scale = np.mean(np.sqrt(np.sum(diffs ** 2, axis=1)))
                    if local_scale > 0.1:
                        length_scales = np.clip(length_scales * 0.9 + local_scale * 0.1, 0.5, 10.0)

            # Occasionally refine around best point with smaller noise
            if calls % 50 == 0 and calls < self.budget - 20:
                # Local refinement
                n_refine = 10
                X_refine = best_x + rng.normal(0, 0.5, (n_refine, self.dim))
                X_refine = np.clip(X_refine, -5, 5)
                for x_r in X_refine:
                    if calls >= self.budget or time.time() > deadline - 2.0:
                        break
                    f_r = func(x_r)
                    calls += 1
                    X = np.vstack([X, x_r])
                    Y = np.append(Y, f_r)
                    if f_r < best_val:
                        best_val = f_r
                        best_x = x_r

        return best_x, best_val
# EVOLVE-BLOCK-END
