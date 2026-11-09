"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Bayesian-Optimization-style surrogate-assisted search for Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit

        rng = np.random.default_rng(12345)

        # Storage for observed points
        X = []  # list of (dim,) arrays
        Y = []  # list of floats

        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal best_x, best_f
            try:
                v = func(np.asarray(x, dtype=float))
                if np.isfinite(v):
                    if v < best_f:
                        best_f = v
                        best_x = np.asarray(x, dtype=float).copy()
                    X.append(np.asarray(x, dtype=float).copy())
                    Y.append(v)
                    return v
            except Exception:
                pass
            return 1e10

        # --- Phase 0: Initial diverse sampling ---
        n_init = 40
        init_budget = min(n_init, budget)
        for i in range(init_budget):
            if time.time() > deadline:
                break
            if i == 0:
                x = np.zeros(dim)
            elif i == 1:
                x = rng.uniform(-5, 5, dim)
            elif i == 2:
                x = rng.uniform(-3, 3, dim)
            elif i == 3:
                x = 4.20968746 * np.ones(dim)  # Known Schwefel minimum coordinate
            elif i == 4:
                x = rng.uniform(-5, 5, dim)
            elif i == 5:
                x = rng.uniform(-4, 4, dim)
            elif i == 6:
                x = rng.uniform(-2, 2, dim)
            elif i == 7:
                x = rng.uniform(-5, 5, dim)
            elif i == 8:
                x = rng.uniform(-3, 3, dim)
            elif i == 9:
                x = rng.uniform(-4, 4, dim)
            elif i == 10:
                x = rng.uniform(-1, 1, dim)
            elif i == 11:
                x = rng.uniform(-5, 5, dim)
            elif i == 12:
                x = rng.uniform(-2.5, 2.5, dim)
            elif i == 13:
                x = rng.uniform(-4.5, 4.5, dim)
            elif i == 14:
                x = rng.uniform(-3.5, 3.5, dim)
            elif i == 15:
                x = rng.uniform(-5, 5, dim)
            elif i == 16:
                x = rng.uniform(-1.5, 1.5, dim)
            elif i == 17:
                x = rng.uniform(-4, 4, dim)
            elif i == 18:
                x = rng.uniform(-2, 2, dim)
            elif i == 19:
                x = rng.uniform(-3, 3, dim)
            elif i == 20:
                x = rng.uniform(-5, 5, dim)
            elif i == 21:
                x = rng.uniform(-2.5, 2.5, dim)
            elif i == 22:
                x = rng.uniform(-4, 4, dim)
            elif i == 23:
                x = rng.uniform(-1, 1, dim)
            elif i == 24:
                x = rng.uniform(-3.5, 3.5, dim)
            elif i == 25:
                x = rng.uniform(-5, 5, dim)
            elif i == 26:
                x = rng.uniform(-2, 2, dim)
            elif i == 27:
                x = rng.uniform(-4.5, 4.5, dim)
            elif i == 28:
                x = rng.uniform(-1.5, 1.5, dim)
            elif i == 29:
                x = rng.uniform(-3, 3, dim)
            elif i == 30:
                x = rng.uniform(-5, 5, dim)
            elif i == 31:
                x = rng.uniform(-2.5, 2.5, dim)
            elif i == 32:
                x = rng.uniform(-4, 4, dim)
            elif i == 33:
                x = rng.uniform(-1, 1, dim)
            elif i == 34:
                x = rng.uniform(-3.5, 3.5, dim)
            elif i == 35:
                x = rng.uniform(-5, 5, dim)
            elif i == 36:
                x = rng.uniform(-2, 2, dim)
            elif i == 37:
                x = rng.uniform(-4.5, 4.5, dim)
            elif i == 38:
                x = rng.uniform(-1.5, 1.5, dim)
            else:
                x = rng.uniform(-3, 3, dim)

            safe_func(x)

        if best_x is None:
            safe_func(np.zeros(dim))
            if best_x is None:
                return np.zeros(dim), 1e10

        # --- Phase 1: Surrogate-guided search ---
        # Use a simple Gaussian Process with Matern kernel
        # We'll implement a lightweight GP for 5D

        def matern_kernel(X1, X2, length_scale, variance):
            """Compute Matern 5/2 kernel between sets of points."""
            X1 = np.asarray(X1)
            X2 = np.asarray(X2)
            d = X1.shape[1]
            # Compute pairwise distances
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            sqrt5_dist = np.sqrt(5.0) * dist / length_scale
            K = variance * (1 + sqrt5_dist + sqrt5_dist ** 2 / 3.0) * np.exp(-sqrt5_dist)
            return K

        def gp_predict(X_obs, Y_obs, X_new, length_scale, variance, noise):
            """GP posterior mean and variance at X_new."""
            K = matern_kernel(X_obs, X_obs, length_scale, variance) + noise * np.eye(len(X_obs))
            K_new = matern_kernel(X_obs, X_new, length_scale, variance)
            try:
                alpha = np.linalg.solve(K, Y_obs)
                mean = K_new.T @ alpha
                K_new_new = matern_kernel(X_new, X_new, length_scale, variance)
                var = np.diag(K_new_new) - np.sum(K_new ** 2 * np.linalg.solve(K, K_new.T.T), axis=0)
                var = np.maximum(var, 1e-12)
                return mean, var
            except np.linalg.LinAlgError:
                # Fallback: use simple heuristic
                dists = np.sqrt(np.sum((X_obs[:, np.newaxis, :] - X_new[np.newaxis, :, :]) ** 2, axis=2))
                weights = np.exp(-dists ** 2 / (2 * length_scale ** 2))
                weights /= weights.sum(axis=0)
                mean = weights.T @ Y_obs
                var = np.ones(len(X_new)) * variance
                return mean, var

        def expected_improvement(mean, var, best_f, xi=0.01):
            """Compute EI acquisition function."""
            sigma = np.sqrt(var)
            z = (best_f - mean - xi) / (sigma + 1e-12)
            ei = (best_f - mean - xi) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        # Adaptive length scale
        length_scale = 2.0
        variance = 100.0
        noise = 1e-4

        remaining_budget = budget - len(Y)
        if remaining_budget <= 0:
            return best_x, best_f

        # We'll use a batch of candidate points and pick the best by EI
        n_candidates = 50
        stagnation_counter = 0
        no_improve_threshold = 30

        for step in range(remaining_budget):
            if time.time() > deadline:
                break

            # Generate candidate points
            candidates = []
            # Mix of: near best, random, and interpolated
            for j in range(n_candidates):
                if j == 0:
                    # Near current best
                    x_cand = best_x + 0.5 * rng.standard_normal(dim)
                elif j == 1:
                    # Pure random
                    x_cand = rng.uniform(-5, 5, dim)
                elif j == 2:
                    # Near best, smaller step
                    x_cand = best_x + 0.2 * rng.standard_normal(dim)
                elif j == 3:
                    # Interpolate between best and a random point
                    x_rand = rng.uniform(-5, 5, dim)
                    t = rng.uniform(0.3, 0.7)
                    x_cand = t * best_x + (1 - t) * x_rand
                elif j < 10:
                    # Cluster around best
                    x_cand = best_x + rng.uniform(-1, 1, dim)
                elif j < 20:
                    # Moderate exploration
                    x_cand = rng.uniform(-4, 4, dim)
                elif j < 30:
                    # Wide exploration
                    x_cand = rng.uniform(-5, 5, dim)
                elif j < 40:
                    # Near best with larger step
                    x_cand = best_x + rng.uniform(-2, 2, dim)
                else:
                    # Random
                    x_cand = rng.uniform(-5, 5, dim)

                x_cand = np.clip(x_cand, -5, 5)
                candidates.append(x_cand)

            candidates = np.array(candidates)

            # Use GP to score candidates
            if len(X) >= 5:
                try:
                    X_obs = np.array(X)
                    Y_obs = np.array(Y)
                    mean, var = gp_predict(X_obs, Y_obs, candidates, length_scale, variance, noise)
                    ei = expected_improvement(mean, var, best_f, xi=0.01)
                    # Pick the candidate with highest EI
                    idx = np.argmax(ei)
                except Exception:
                    # Fallback: pick randomly
                    idx = rng.integers(n_candidates)
            else:
                idx = rng.integers(n_candidates)

            x_new = candidates[idx]
            f_new = safe_func(x_new)

            # Update length scale adaptively
            if f_new < best_f:
                stagnation_counter = 0
                length_scale = max(0.5, length_scale * 0.9)
            else:
                stagnation_counter += 1
                length_scale = min(5.0, length_scale * 1.05)

            # If stagnating, do a random restart
            if stagnation_counter >= no_improve_threshold:
                stagnation_counter = 0
                x_restart = rng.uniform(-5, 5, dim)
                safe_func(x_restart)
                length_scale = 3.0

        # --- Phase 2: Local refinement with Nelder-Mead ---
        if time.time() < deadline - 5 and best_x is not None:
            try:
                result = minimize(
                    lambda x: safe_func(x),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.fun < best_f:
                    best_x = result.x
                    best_f = result.fun
            except Exception:
                pass

        # --- Phase 3: Final polish with small steps ---
        if time.time() < deadline - 2 and best_x is not None:
            sigma = 0.1
            for _ in range(20):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                if fy < best_f:
                    sigma *= 0.9
                else:
                    sigma *= 0.8
                if sigma < 1e-8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
