"""Surrogate-assisted optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Surrogate-assisted search with local refinement for Discus."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin for 300s limit
        calls = 0

        def safe_call(x):
            nonlocal calls
            if calls >= self.budget:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (self.dim,):
                x = x.reshape(self.dim)
            fx = func(x)
            calls += 1
            if not np.isfinite(fx):
                return np.inf
            return float(fx)

        # Phase 1: Initial exploration with random sampling + local refinement
        rng = np.random.default_rng(42)
        x_best = None
        f_best = np.inf
        candidates = []

        # Random initial points
        n_initial = 20
        for i in range(n_initial):
            x = rng.uniform(-5, 5, self.dim)
            f = safe_call(x)
            candidates.append((x, f))
            if f < f_best:
                f_best = f
                x_best = x.copy()

        # Local refinement around best point using Nelder-Mead
        if x_best is not None and calls < self.budget:
            try:
                result = minimize(
                    lambda x: safe_call(x),
                    x_best,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10,
                             'adaptive': True, 'initial_simplex': 0.5}
                )
                if result.fun < f_best and np.isfinite(result.fun):
                    f_best = result.fun
                    x_best = result.x.copy()
            except Exception:
                pass

        # Phase 2: Surrogate-assisted search using Gaussian Process
        # Build training set from all evaluations so far
        X = np.array([c[0] for c in candidates])
        Y = np.array([c[1] for c in candidates])

        # Filter out inf values
        mask = np.isfinite(Y)
        X = X[mask]
        Y = Y[mask]

        if len(X) < 5:
            # Not enough data, continue with random search
            for _ in range(self.budget - calls):
                if time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, self.dim)
                f = safe_call(x)
                if f < f_best:
                    f_best = f
                    x_best = x.copy()
            return x_best, f_best

        # Define GP-based acquisition (Expected Improvement)
        def gp_predict(X_new):
            """Simple GP prediction using RBF kernel."""
            n = len(X)
            # Compute kernel matrix
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    diff = X[i] - X[j]
                    K[i, j] = np.exp(-0.5 * np.sum(diff ** 2) / 1.0)
            K += 1e-6 * np.eye(n)  # regularization

            # Predict at X_new
            m = len(X_new)
            K_star = np.zeros((m, n))
            for i in range(m):
                for j in range(n):
                    diff = X_new[i] - X[j]
                    K_star[i, j] = np.exp(-0.5 * np.sum(diff ** 2) / 1.0)

            # Solve for weights
            try:
                alpha = np.linalg.solve(K, Y)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, Y, rcond=None)[0]

            # Mean prediction
            mean = K_star @ alpha

            # Variance prediction
            var = np.ones(m) - np.sum(K_star * alpha[None, :], axis=1)
            var = np.maximum(var, 1e-8)

            return mean, np.sqrt(var)

        # Expected Improvement acquisition
        def ei_acquisition(X_new):
            mean, std = gp_predict(X_new)
            improvement = f_best - mean
            z = improvement / (std + 1e-8)
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            return -ei  # minimize negative EI

        # Phase 2 iterations
        n_surrogate_iters = 50
        for iter_idx in range(n_surrogate_iters):
            if calls >= self.budget or time.time() > deadline:
                break

            # Sample candidate points for acquisition
            n_candidates = 200
            X_cand = rng.uniform(-5, 5, (n_candidates, self.dim))

            # Evaluate acquisition function
            scores = ei_acquisition(X_cand)
            idx = np.argmin(scores)  # most negative = highest EI
            x_next = X_cand[idx].copy()

            # Evaluate at best point
            f_next = safe_call(x_next)
            candidates.append((x_next, f_next))

            if f_next < f_best:
                f_best = f_next
                x_best = x_next.copy()

                # Local refinement around new best
                if calls < self.budget - 10:
                    try:
                        result = minimize(
                            lambda x: safe_call(x),
                            x_best,
                            method='Nelder-Mead',
                            options={'maxiter': 30, 'xatol': 1e-8, 'fatol': 1e-10,
                                     'adaptive': True, 'initial_simplex': 0.3}
                        )
                        if result.fun < f_best and np.isfinite(result.fun):
                            f_best = result.fun
                            x_best = result.x.copy()
                    except Exception:
                        pass

            # Update GP training data (keep last 50 points for efficiency)
            if len(candidates) > 50:
                candidates = candidates[-50:]

            # Rebuild X, Y from candidates
            X = np.array([c[0] for c in candidates])
            Y = np.array([c[1] for c in candidates])
            mask = np.isfinite(Y)
            X = X[mask]
            Y = Y[mask]

            if len(X) < 3:
                break

        # Phase 3: Final local refinement
        if x_best is not None and calls < self.budget - 5 and time.time() < deadline:
            try:
                result = minimize(
                    lambda x: safe_call(x),
                    x_best,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12,
                             'adaptive': True, 'initial_simplex': 0.1}
                )
                if result.fun < f_best and np.isfinite(result.fun):
                    f_best = result.fun
                    x_best = result.x.copy()
            except Exception:
                pass

        return x_best, f_best
# EVOLVE-BLOCK-END
