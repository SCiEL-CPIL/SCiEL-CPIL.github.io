"""Bayesian Optimization with GP surrogate for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimization with Gaussian Process surrogate."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial random samples to bootstrap the GP
        n_init = min(10, budget)
        X = rng.uniform(-5, 5, size=(n_init, dim))
        Y = np.array([func(x) for x in X])

        # GP hyperparameters
        # We'll use a simple ARD Matern 5/2 kernel
        # Length scales initialized to 1.0, signal variance to 1.0
        # Noise variance small

        # We need to be careful with time. 500 calls.
        # GP fitting is O(n^3). With n=500, that's 125M ops per fit.
        # If we refit every call, it might be too slow in Python/NumPy without optimization.
        # However, scipy's minimize is also expensive.
        
        # Strategy:
        # 1. Use a subset of points for GP fitting if n gets large? Or just use all?
        #    With dim=5, n=500, O(n^3) is 125M. NumPy can do this in <1s usually.
        #    But we do it 500 times? 500 * 1s = 500s. Too slow.
        #    We need to be smarter.
        #    Maybe fit GP every few steps? Or use a simpler surrogate?
        #    Or use a subset of the most recent points?
        #    Or use a fixed number of points for GP (e.g., last 50)?
        
        # Let's try using all points but optimize the GP implementation.
        # Actually, 500 calls * O(500^3) is likely too slow.
        # Let's limit the GP training set to the last 30-50 points or use a subset.
        # Or, use a cheaper acquisition function optimization.
        
        # Alternative: Use a simpler surrogate like Kriging with a fixed kernel,
        # or just use Random Forest? No, GP is requested.
        
        # Let's try to make the GP fast.
        # We can use the Cholesky decomposition.
        # If it's too slow, we can reduce the training set size.
        
        # Let's start with a modest training set size, say 20 points, updated with the best.
        # Actually, for BO, using all points is standard but expensive.
        # Let's try using all points and see if it fits in time.
        # If not, we'll need to optimize.
        
        # Given the 300s limit, and 500 calls, we have 0.6s per call.
        # GP fit + Acquisition optimization must fit in 0.6s.
        # O(n^3) for n=500 is ~125M flops. NumPy is fast, maybe 0.1-0.5s.
        # Acquisition optimization: minimize LCB. This requires evaluating the GP at many points.
        # Each GP evaluation is O(n^2) for prediction (if we precompute Cholesky).
        # If we do 100 evaluations of the acquisition function, that's 100 * O(n^2) = 100 * 250k = 25M.
        # So total ~150M flops. Should be okay.
        
        # Let's implement a simple GP.
        
        def matern52_kernel(X1, X2, lengthscales, signal_var):
            # X1: (n, d), X2: (m, d)
            # lengthscales: (d,)
            X1_scaled = X1 / lengthscales
            X2_scaled = X2 / lengthscales
            # Squared distances
            # Use broadcasting
            sq_dists = np.sum(X1_scaled**2, axis=1)[:, None] + np.sum(X2_scaled**2, axis=1)[None, :] - 2 * X1_scaled @ X2_scaled.T
            sq_dists = np.maximum(sq_dists, 0)
            r = np.sqrt(sq_dists)
            kernel = signal_var * (1 + np.sqrt(5) * r + 5/3 * r**2) * np.exp(-np.sqrt(5) * r)
            return kernel

        def predict_gp(X_train, y_train, X_test, lengthscales, signal_var, noise_var):
            K = matern52_kernel(X_train, X_train, lengthscales, signal_var) + noise_var * np.eye(len(X_train))
            K += 1e-8 * np.eye(len(X_train))  # jitter
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            K_s = matern52_kernel(X_test, X_train, lengthscales, signal_var)
            mu = K_s @ alpha
            v = np.linalg.solve(L, K_s.T)
            var = signal_var - np.sum(v**2, axis=1)
            var = np.maximum(var, 1e-8)
            return mu, var

        def lcb_acquisition(X, X_train, y_train, lengthscales, signal_var, noise_var, xi=2.576):
            mu, var = predict_gp(X_train, y_train, X, lengthscales, signal_var, noise_var)
            sigma = np.sqrt(var)
            return mu - xi * sigma

        # Initial hyperparameters
        lengthscales = np.ones(dim)
        signal_var = 1.0
        noise_var = 1e-4

        # We might want to adapt hyperparameters, but that adds complexity.
        # Let's start with fixed hyperparameters and see.
        
        # For each step:
        # 1. Fit GP (or use cached Cholesky if possible, but X changes)
        # 2. Optimize acquisition function
        # 3. Evaluate new point
        
        # To speed up acquisition optimization, we can use a multi-start local optimization.
        
        for step in range(n_init, budget):
            # Fit GP
            # If n is large, this is expensive.
            n_train = len(X)
            
            # Optimization of acquisition function
            # We'll use scipy minimize on the LCB
            # We need to define the acquisition function to be minimized (negative LCB? No, LCB is to be minimized for minimization)
            # Wait, LCB = mu - xi * sigma. We want to minimize this.
            
            def acq_func(x):
                x = x.reshape(1, -1)
                x = np.clip(x, -5, 5)
                val = lcb_acquisition(x, X, Y, lengthscales, signal_var, noise_var)
                return val[0]
            
            # Multi-start
            best_x = None
            best_val = np.inf
            
            # Start from current best and random points
            starts = [X[np.argmin(Y)]]
            starts += [rng.uniform(-5, 5, dim) for _ in range(5)]
            
            for x0 in starts:
                res = minimize(acq_func, x0, method='L-BFGS-B', bounds=[(-5, 5)] * dim, options={'maxiter': 50})
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            
            if best_x is None:
                best_x = rng.uniform(-5, 5, dim)
            
            # Evaluate
            best_x = np.clip(best_x, -5, 5)
            y_new = func(best_x)
            
            # Update X, Y
            X = np.vstack([X, best_x])
            Y = np.append(Y, y_new)
            
            # Optional: Update hyperparameters? 
            # For now, keep fixed.
            
        return X[np.argmin(Y)], np.min(Y)
# EVOLVE-BLOCK-END
