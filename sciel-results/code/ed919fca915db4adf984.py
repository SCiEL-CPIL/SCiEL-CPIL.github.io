"""Bayesian Optimization with Gaussian Process surrogate for BBOB f14."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimization using a Gaussian Process surrogate."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Latin Hypercube sampling for initial points
        n_initial = min(20, budget // 5)
        if n_initial < 5:
            n_initial = 5

        # Generate LHS points
        x_init = np.zeros((n_initial, dim))
        for d in range(dim):
            # Permute indices for LHS
            idx = rng.permutation(n_initial)
            x_init[idx, d] = (idx + 0.5) / n_initial
        # Scale to [-5, 5]
        x_init = x_init * 10.0 - 5.0

        # Evaluate initial points
        X = x_init.copy()
        Y = np.array([func(x) for x in X])
        n_calls = n_initial

        # GP parameters
        # We use a simple GP with Matern kernel (nu=2.5) for smoothness
        # To keep it fast, we'll use a simplified approach:
        # - Standardize inputs
        # - Use a fixed length-scale initially, adapt if needed
        # - For 500 calls in 5D, we need efficient GP operations

        # Precompute standardization
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8

        def standardize(x):
            return (x - X_mean) / X_std

        # Kernel function: Matern 5/2
        def matern_kernel(X1, X2, length_scale, variance):
            """Compute Matern 5/2 kernel between two sets of points."""
            X1_std = standardize(X1)
            X2_std = standardize(X2)
            # Pairwise distances
            diff = X1_std[:, np.newaxis, :] - X2_std[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
            # Matern 5/2
            sqrt5_d = np.sqrt(5) * dist / length_scale
            K = variance * (1 + sqrt5_d + sqrt5_d ** 2 / 3.0) * np.exp(-sqrt5_d)
            return K

        # Hyperparameters
        length_scale = 1.0
        variance = 1.0
        noise_var = 1e-6

        # We'll adapt hyperparameters coarsely every 50 evaluations
        # For now, use fixed reasonable values

        best_x = X[np.argmin(Y)]
        best_y = Y.min()

        # Acquisition: Expected Improvement
        def expected_improvement(X_test, y_mean, y_std, f_best, xi=0.01):
            """Compute EI for a set of candidate points."""
            y_best = f_best - xi
            z = (y_best - y_mean) / (y_std + 1e-12)
            ei = (y_best - y_mean) * norm.cdf(z) + (y_std + 1e-12) * norm.pdf(z)
            ei = np.maximum(ei, 0.0)
            return ei

        # GP prediction function
        def gp_predict(X_test):
            """Predict mean and variance at X_test given training data X, Y."""
            X_test_std = standardize(X_test)
            K_train = matern_kernel(X, X, length_scale, variance) + noise_var * np.eye(len(X))
            K_test = matern_kernel(X, X_test, length_scale, variance)
            # Solve for weights
            try:
                K_inv_y = np.linalg.solve(K_train, Y)
            except np.linalg.LinAlgError:
                K_inv_y = np.linalg.lstsq(K_train, Y, rcond=None)[0]
            mu = K_test.T @ K_inv_y
            K_test_test = matern_kernel(X_test, X_test, length_scale, variance)
            var = np.diag(K_test_test) - np.sum(K_test ** 2 * K_inv_y[np.newaxis, :], axis=1)
            var = np.maximum(var, 1e-12)
            return mu, np.sqrt(var)

        # Main optimization loop
        while n_calls < budget:
            # Generate candidate points for acquisition
            # Use a mix of random and structured candidates
            n_candidates = 200
            X_cand = rng.uniform(-5, 5, (n_candidates, dim))

            # Predict mean and std at candidates
            mu, std = gp_predict(X_cand)

            # Compute EI
            ei = expected_improvement(X_cand, mu, std, best_y)

            # Select best candidate
            idx = np.argmax(ei)
            x_new = X_cand[idx]
            y_new = func(x_new)
            n_calls += 1

            # Update training set
            X = np.vstack([X, x_new])
            Y = np.append(Y, y_new)

            # Update best
            if y_new < best_y:
                best_y = y_new
                best_x = x_new

            # Periodically update standardization and hyperparameters
            if n_calls % 50 == 0:
                X_mean = X.mean(axis=0)
                X_std = X.std(axis=0) + 1e-8

            # Optionally adapt length_scale based on improvement
            if n_calls % 100 == 0 and n_calls > n_initial:
                # Simple adaptation: if we're making progress, shrink LS
                # This is a heuristic; keep it simple
                pass

        return best_x, best_y

# EVOLVE-BLOCK-END
