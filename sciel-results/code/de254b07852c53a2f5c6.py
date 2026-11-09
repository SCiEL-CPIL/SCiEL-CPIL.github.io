"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm
import math


class Optimizer:
    """Bayesian Optimization with GP surrogate and Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial random sampling
        n_init = min(10, budget)
        X = rng.uniform(-5, 5, (n_init, dim))
        Y = np.array([func(x) for x in X])

        best_x = X[np.argmin(Y)]
        best_y = Y.min()

        # GP kernel parameters
        length_scale = 2.0
        noise = 1e-6
        variance = 1.0

        # Normalize Y for numerical stability
        y_mean = Y.mean()
        y_std = max(Y.std(), 1e-8)

        def gp_predict(X_new, X_train, Y_train, ls, var, noise):
            """Compute GP mean and variance at X_new given training data."""
            Y_train_norm = (Y_train - y_mean) / y_std
            K = self._kernel(X_train, X_train, ls, var) + noise * np.eye(len(X_train))
            K_new = self._kernel(X_train, X_new, ls, var)
            try:
                L = np.linalg.cholesky(K + 1e-10 * np.eye(len(K)))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train_norm))
                mu = K_new.T @ alpha * y_std + y_mean
                v = np.linalg.solve(L, K_new)
                var_pred = var - (K_new.T @ v)
                var_pred = np.maximum(var_pred, 1e-10)
                return mu, np.sqrt(var_pred)
            except np.linalg.LinAlgError:
                # Fallback: use simple distance-based heuristic
                dists = np.linalg.norm(X_new[:, None, :] - X_train[None, :, :], axis=2)
                min_dists = dists.min(axis=1)
                mu = Y_train[np.argmin(dists, axis=1)]
                var_pred = np.ones(len(X_new)) * 1.0
                return mu, np.sqrt(var_pred)

        def _kernel(self, X1, X2, ls, var):
            """Squared exponential kernel."""
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return var * np.exp(-sq_dists / (2 * ls**2))

        def expected_improvement(X_new, X_train, Y_train, ls, var, noise):
            """Compute EI acquisition function."""
            mu, sigma = gp_predict(X_new, X_train, Y_train, ls, var, noise)
            best_y_current = Y_train.min()
            # For minimization: EI = (best - mu) * Phi((best - mu)/sigma) + sigma * phi((best - mu)/sigma)
            z = (best_y_current - mu) / sigma
            ei = (best_y_current - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        calls_used = n_init
        remaining = budget - n_init

        # Adaptive phase: alternate between BO and local refinement
        bo_phase = True
        stagnation_count = 0
        no_improve_threshold = 15

        for i in range(remaining):
            # Check deadline
            if time.time() > deadline - 5.0:
                break

            if bo_phase:
                # Bayesian Optimization step
                # Sample candidate points
                n_candidates = min(200, max(50, budget - calls_used))
                X_cand = rng.uniform(-5, 5, (n_candidates, dim))

                # Also include points near current best
                n_near = min(50, n_candidates // 4)
                X_near = best_x + rng.normal(0, 0.5, (n_near, dim))
                X_near = np.clip(X_near, -5, 5)
                X_cand = np.vstack([X_cand, X_near])

                # Compute EI
                ei_vals = expected_improvement(X_cand, X, Y, length_scale, variance, noise)

                # Select best candidate
                idx = np.argmax(ei_vals)
                x_new = X_cand[idx]

                # Evaluate
                y_new = func(x_new)
                calls_used += 1

                # Update training set
                X = np.vstack([X, x_new])
                Y = np.append(Y, y_new)

                if y_new < best_y - 1e-10:
                    best_x = x_new
                    best_y = y_new
                    stagnation_count = 0
                else:
                    stagnation_count += 1

                # Adapt length scale based on progress
                if stagnation_count > no_improve_threshold:
                    # Switch to local refinement
                    bo_phase = False
                    length_scale *= 0.5
                    if length_scale < 0.3:
                        # Restart with new random point
                        x_restart = rng.uniform(-5, 5, dim)
                        y_restart = func(x_restart)
                        calls_used += 1
                        X = np.array([x_restart])
                        Y = np.array([y_restart])
                        if y_restart < best_y:
                            best_x = x_restart
                            best_y = y_restart
                        bo_phase = True
                        stagnation_count = 0
                        length_scale = 2.0
            else:
                # Local refinement around best point using local search
                n_local = min(20, max(5, budget - calls_used))
                X_local = best_x + rng.normal(0, 0.3, (n_local, dim))
                X_local = np.clip(X_local, -5, 5)

                best_local_x = best_x
                best_local_y = best_y

                for x_l in X_local:
                    if time.time() > deadline - 5.0:
                        break
                    y_l = func(x_l)
                    calls_used += 1
                    if y_l < best_local_y:
                        best_local_x = x_l
                        best_local_y = y_l

                if best_local_y < best_y - 1e-10:
                    best_x = best_local_x
                    best_y = best_local_y
                    stagnation_count = 0
                else:
                    stagnation_count += 1

                # Return to BO if local search stagnant
                if stagnation_count > 10:
                    bo_phase = True
                    length_scale = max(1.0, length_scale * 1.5)
                    # Add local best to training set if not already there
                    if len(X) < 200:
                        X = np.vstack([X, best_x])
                        Y = np.append(Y, best_y)
                    stagnation_count = 0

        return best_x, best_y
# EVOLVE-BLOCK-END
