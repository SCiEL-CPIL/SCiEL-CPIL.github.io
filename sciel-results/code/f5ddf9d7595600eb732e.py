"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-fidelity sequential model-based optimization with GP surrogate."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # State
        X = np.zeros((0, dim))
        Y = np.zeros(0)
        best_x = None
        best_y = np.inf

        # GP hyperparameters (log scale)
        # We'll use a simple GP with RBF kernel and noise
        # For efficiency, we'll use a simplified approach:
        # - Use a small number of initial random points
        # - Fit a simple quadratic or use scipy's minimize on a surrogate

        # Phase 1: Initial random sampling
        n_init = min(20, budget)
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            X = np.vstack([X, x]) if X.size > 0 else x.reshape(1, -1)
            Y = np.append(Y, fx)
            if fx < best_y:
                best_y = fx
                best_x = x.copy()

        remaining = budget - n_init
        if remaining <= 0:
            return best_x, best_y

        # Phase 2: Iterative surrogate-based optimization
        # Use a simple GP-like approach with expected improvement
        # For speed, we'll use a quadratic surrogate fit to recent points

        # We'll alternate between:
        # 1. Fitting a local quadratic model
        # 2. Finding its minimum
        # 3. Evaluating there

        # Use a subset of points for the surrogate (most recent + best)
        # This keeps the surrogate fit fast

        n_surrogate = min(50, len(Y))

        for iteration in range(remaining):
            if time.time() > deadline:
                break

            # Select points for surrogate: best point + most recent points
            if len(Y) <= n_surrogate:
                idx = np.arange(len(Y))
            else:
                # Take best point and most recent n_surrogate-1 points
                best_idx = np.argmin(Y)
                recent_idx = np.arange(len(Y) - n_surrogate + 1, len(Y))
                # Combine and remove duplicates
                idx = np.unique(np.concatenate([[best_idx], recent_idx]))
                if len(idx) > n_surrogate:
                    idx = idx[-n_surrogate:]

            Xs = X[idx]
            Ys = Y[idx]

            # Fit a quadratic surrogate
            # Use least squares to fit: f(x) ≈ c0 + c1*x + 0.5*c2*x^2 (diagonal quadratic)
            # For simplicity, use a linear + diagonal quadratic model
            # Features: [1, x1, ..., xd, x1^2, ..., xd^2]
            n_pts = len(Ys)
            n_features = 1 + dim + dim  # intercept + linear + quadratic diagonal

            if n_pts >= n_features:
                # Build feature matrix
                A = np.ones((n_pts, n_features))
                A[:, 1:1 + dim] = Xs
                A[:, 1 + dim:] = Xs ** 2

                # Solve least squares
                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(A, Ys, rcond=None)
                except:
                    coeffs = np.zeros(n_features)

                # Find minimum of the quadratic surrogate
                # The quadratic form is: c0 + c1^T x + 0.5 * c2^T x^2
                # Gradient: c1 + c2 * x = 0 => x* = -c1/c2 (element-wise if diagonal)
                c1 = coeffs[1:1 + dim]
                c2 = coeffs[1 + dim:]

                # Only use where c2 > 0 (convex direction)
                valid = c2 > 1e-10
                if np.any(valid):
                    x_star = np.zeros(dim)
                    x_star[valid] = -c1[valid] / c2[valid]
                    # Clip to bounds
                    x_star = np.clip(x_star, -5, 5)
                else:
                    # If no convex direction, use gradient descent step from best
                    x_star = best_x - 0.5 * c1
                    x_star = np.clip(x_star, -5, 5)

                # Add some exploration
                if rng.random() < 0.3:
                    x_star = x_star + 0.5 * rng.standard_normal(dim)
                    x_star = np.clip(x_star, -5, 5)
            else:
                # Not enough points, do random exploration near best
                x_star = best_x + 1.0 * rng.standard_normal(dim)
                x_star = np.clip(x_star, -5, 5)

            # Evaluate
            fx = func(x_star)

            # Update state
            X = np.vstack([X, x_star])
            Y = np.append(Y, fx)

            if fx < best_y:
                best_y = fx
                best_x = x_star.copy()

        return best_x, best_y
# EVOLVE-BLOCK-END
