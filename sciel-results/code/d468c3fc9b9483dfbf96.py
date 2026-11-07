"""Bayesian Optimization surrogate for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimization with Gaussian Process surrogate."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _rbf_kernel(self, X1, X2, length_scale, variance):
        """Compute RBF kernel matrix between two sets of points."""
        # X1: (n1, d), X2: (n2, d)
        # Squared distances
        sq1 = np.sum(X1 ** 2, axis=1)[:, np.newaxis]
        sq2 = np.sum(X2 ** 2, axis=1)[np.newaxis, :]
        sqdist = sq1 + sq2 - 2 * X1 @ X2.T
        sqdist = np.maximum(sqdist, 0)  # numerical stability
        return variance * np.exp(-0.5 * sqdist / (length_scale ** 2))

    def _fit_gp_params(self, X, y, rng):
        """Fit GP hyperparameters via simple optimization."""
        # y: (n,)
        n = len(y)
        
        # Standardize y for better conditioning
        y_mean = np.mean(y)
        y_std = np.std(y) + 1e-8
        y_norm = (y - y_mean) / y_std
        
        # Initial hyperparameters
        length_scale = 1.0
        variance = 1.0
        noise = 0.01
        
        # Simple coordinate descent / grid search for hyperparameters
        best_neg_loglik = np.inf
        best_params = (length_scale, variance, noise)
        
        # Try a few candidate length scales
        for ls in [0.5, 1.0, 1.5, 2.0, 3.0]:
            for var in [0.5, 1.0, 2.0]:
                for noise in [0.001, 0.01, 0.1]:
                    try:
                        K = self._rbf_kernel(X, X, ls, var) + noise * np.eye(n)
                        # Add small jitter for numerical stability
                        K += 1e-6 * np.eye(n)
                        
                        # Compute negative log likelihood
                        L = np.linalg.cholesky(K)
                        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                        logdet = 2 * np.sum(np.log(np.diag(L)))
                        neg_loglik = 0.5 * y_norm @ alpha + 0.5 * logdet + 0.5 * n * np.log(2 * np.pi)
                        
                        if neg_loglik < best_neg_loglik:
                            best_neg_loglik = neg_loglik
                            best_params = (ls, var, noise)
                    except:
                        continue
        
        return best_params

    def _gp_predict(self, X, X_train, y_train, length_scale, variance, noise):
        """Predict mean and variance at X given training data."""
        n = len(y_train)
        y_mean = np.mean(y_train)
        y_std = np.std(y_train) + 1e-8
        y_norm = (y_train - y_mean) / y_std
        
        K = self._rbf_kernel(X_train, X_train, length_scale, variance) + noise * np.eye(n)
        K += 1e-6 * np.eye(n)
        K_star = self._rbf_kernel(X, X_train, length_scale, variance)
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            v = np.linalg.solve(L, K_star.T)
            K_ss = self._rbf_kernel(X, X, length_scale, variance)
            
            mu_norm = K_star @ alpha
            var = np.diag(K_ss) - np.sum(v ** 2, axis=1)
            var = np.maximum(var, 1e-8)
            
            mu = mu_norm * y_std + y_mean
            std = np.sqrt(var) * y_std
        except:
            # Fallback: uniform prediction
            mu = np.full(X.shape[0], y_mean)
            std = np.full(X.shape[0], y_std)
        
        return mu, std

    def _ei_acquisition(self, X, X_train, y_train, length_scale, variance, noise, f_min):
        """Expected Improvement acquisition function."""
        mu, std = self._gp_predict(X, X_train, y_train, length_scale, variance, noise)
        improvement = f_min - mu
        with np.errstate(divide='ignore', invalid='ignore'):
            z = improvement / std
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            ei = np.where(std > 1e-8, ei, 0)
        return ei

    def _optimize_acquisition(self, X_train, y_train, length_scale, variance, noise, f_min, rng, n_restarts=10):
        """Maximize EI acquisition function via multi-start local optimization."""
        best_x = None
        best_ei = -np.inf
        
        # Start from training points and random points
        starts = []
        for x in X_train:
            starts.append(x.copy())
        for _ in range(n_restarts - len(X_train)):
            starts.append(rng.uniform(-5, 5, self.dim))
        
        for x0 in starts:
            x0 = np.clip(x0, -5, 5)
            
            def neg_ei(x):
                x = np.clip(x, -5, 5)
                ei = self._ei_acquisition(x.reshape(1, -1), X_train, y_train, length_scale, variance, noise, f_min)
                return -ei[0]
            
            res = minimize(neg_ei, x0, method='Nelder-Mead', 
                          options={'xatol': 0.05, 'fatol': 1e-4, 'maxiter': 200})
            
            if res.success or res.fun < -best_ei:
                x_opt = np.clip(res.x, -5, 5)
                ei_val = -res.fun
                if ei_val > best_ei:
                    best_ei = ei_val
                    best_x = x_opt
        
        if best_x is None:
            best_x = rng.uniform(-5, 5, self.dim)
        
        return best_x

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Initialize with a few random points
        X_train = []
        y_train = []
        
        # Initial random samples
        n_initial = min(10, budget)
        for i in range(n_initial):
            x = rng.uniform(-5, 5, dim)
            y = func(x)
            X_train.append(x)
            y_train.append(y)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        f_min = np.min(y_train)
        best_x = X_train[np.argmin(y_train)]
        
        calls_made = n_initial
        
        # Fit initial GP
        ls, var, noise = self._fit_gp_params(X_train, y_train, rng)
        
        # Bayesian Optimization loop
        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Find next point to evaluate
            x_next = self._optimize_acquisition(X_train, y_train, ls, var, noise, f_min, rng, n_restarts=8)
            
            # Evaluate
            y_next = func(x_next)
            calls_made += 1
            
            # Update training data
            X_train = np.vstack([X_train, x_next.reshape(1, -1)])
            y_train = np.append(y_train, y_next)
            
            # Update best
            if y_next < f_min:
                f_min = y_next
                best_x = x_next
            
            # Refit GP periodically (every 10 new points or so)
            if len(y_train) % 10 == 0:
                try:
                    ls, var, noise = self._fit_gp_params(X_train, y_train, rng)
                except:
                    pass
        
        return best_x, f_min
# EVOLVE-BLOCK-END
