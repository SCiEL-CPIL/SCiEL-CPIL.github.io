"""Optimizer for BBOB f15 (Rastrigin rotated) with surrogate-guided search and ES refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """Surrogate-assisted optimizer with RBF GP and ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280.0
        rng = np.random.default_rng(42)
        dim = self.dim

        best_x = None
        best_f = np.inf
        total_budget_used = 0

        # Phase 1: Random exploration to gather initial samples
        n_explore = 30
        points = []
        values = []

        for i in range(n_explore):
            if total_budget_used >= self.budget:
                break
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            total_budget_used += 1
            points.append(x)
            values.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        points = np.array(points)
        values = np.array(values)

        # Phase 2: Surrogate-assisted optimization
        def rbf_kernel(X1, X2, length_scale=1.0):
            """Compute RBF kernel matrix between two sets of points."""
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-0.5 * sq_dists / (length_scale ** 2))

        def predict_gp(X_new, X_train, y_train, length_scale=1.5, noise=0.1):
            """GP prediction using RBF kernel."""
            K = rbf_kernel(X_train, X_train, length_scale)
            K = K + noise * np.eye(len(X_train))
            Ks = rbf_kernel(X_new, X_train, length_scale)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K = K + 1e-6 * np.eye(len(X_train))
                L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            K_ss = rbf_kernel(X_new, X_new, length_scale)
            mean = Ks @ alpha
            var = np.diag(K_ss - Ks @ np.linalg.solve(L, np.linalg.solve(L.T, Ks.T)))
            var = np.maximum(var, 1e-8)
            return mean, np.sqrt(var)

        def expected_improvement(mean, std, best_val, xi=0.01):
            """Compute Expected Improvement acquisition function."""
            imp = best_val - mean - xi
            with np.errstate(divide='ignore', invalid='ignore'):
                z = imp / (std + 1e-10)
                ei = imp * norm.cdf(z) + std * norm.pdf(z)
            ei = np.where(std > 1e-6, ei, 0.0)
            return ei

        # Surrogate-guided phase
        surrogate_evals = 0
        max_surrogate_evals = self.budget // 3

        while total_budget_used < self.budget and surrogate_evals < max_surrogate_evals:
            if time.time() > deadline:
                break

            if len(points) < 2:
                x = rng.uniform(-5, 5, dim)
            else:
                points_arr = np.array(points)
                values_arr = np.array(values)

                # Fit GP
                try:
                    K = rbf_kernel(points_arr, points_arr, 1.5)
                    K = K + 0.1 * np.eye(len(points_arr))
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    break

                # Sample candidate points
                n_candidates = 200
                candidates = rng.uniform(-5, 5, (n_candidates, dim))

                # Predict GP at candidates
                try:
                    Ks = rbf_kernel(candidates, points_arr, 1.5)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, values_arr))
                    mean = Ks @ alpha
                    K_ss = rbf_kernel(candidates, candidates, 1.5)
                    var = np.diag(K_ss - Ks @ np.linalg.solve(L, np.linalg.solve(L.T, Ks.T)))
                    var = np.maximum(var, 1e-8)
                    std = np.sqrt(var)

                    # Compute EI
                    ei = expected_improvement(mean, std, best_f)
                    idx = np.argmax(ei)
                    x = candidates[idx]
                except Exception:
                    x = rng.uniform(-5, 5, dim)

            fx = func(x)
            total_budget_used += 1
            surrogate_evals += 1
            points.append(x)
            values.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 3: Local refinement with (1+1)-ES around best point
        if best_x is not None and total_budget_used < self.budget:
            remaining = self.budget - total_budget_used
            if remaining > 5:
                x = best_x.copy()
                fx = best_f
                sigma = 0.5
                for _ in range(remaining):
                    if time.time() > deadline:
                        break
                    if total_budget_used >= self.budget:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    total_budget_used += 1
                    if fy < fx:
                        x, fx = y, fy
                        sigma = min(sigma * 1.1, 3.0)
                    else:
                        sigma *= 0.95
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    if sigma < 1e-4:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
