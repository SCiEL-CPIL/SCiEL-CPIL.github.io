"""Bayesian Optimization with GP surrogate for BBOB f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.stats as stats
from scipy.optimize import minimize


class Optimizer:
    """Bayesian Optimization with GP surrogate for BBOB f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        n_calls = 0
        
        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return best_f
            if check_time():
                return best_f
            val = func(x)
            n_calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Initialize with random points to cover the space
        n_init = min(10, budget)
        X_init = np.random.uniform(-5, 5, (n_init, dim))
        y_init = np.array([call_func(x) for x in X_init])
        
        X = X_init
        y = y_init
        
        # GP Parameters
        # Matern 5/2 kernel
        def matern52_kernel(x1, x2, l, sigma_f):
            dist = np.sqrt(np.sum((x1[:, np.newaxis, :] - x2[np.newaxis, :, :])**2, axis=2))
            return sigma_f**2 * (1 + np.sqrt(5)*dist/l + 5*dist**2/(3*l**2)) * np.exp(-np.sqrt(5)*dist/l)

        def log_likelihood(params):
            # params: [log(l), log(sigma_f)]
            try:
                l = np.exp(params[0])
                sigma_f = np.exp(params[1])
                K = matern52_kernel(X, X, l, sigma_f)
                K += 1e-8 * np.eye(len(X))
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L, y)
                alpha = np.linalg.solve(L.T, alpha)
                ll = -0.5 * np.dot(y, alpha) - np.sum(np.log(np.diag(L))) - 0.5 * len(X) * np.log(2 * np.pi)
                return -ll
            except:
                return 1e10

        # Optimization of hyperparameters
        # We will update hyperparameters every few iterations to save time
        current_hyperparams = np.array([np.log(2.0), np.log(1.0)]) # Initial guesses
        
        for i in range(budget - n_init):
            if check_time() or n_calls >= budget:
                break
            
            # Update hyperparameters every 5 iterations or initially
            if i % 5 == 0:
                # Optimize hyperparameters
                res = minimize(log_likelihood, current_hyperparams, method='L-BFGS-B', bounds=[(-2, 3), (-2, 3)])
                if res.success:
                    current_hyperparams = res.x
            
            l = np.exp(current_hyperparams[0])
            sigma_f = np.exp(current_hyperparams[1])
            
            # Compute GP predictions
            K = matern52_kernel(X, X, l, sigma_f)
            K += 1e-8 * np.eye(len(X))
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L, y)
            alpha = np.linalg.solve(L.T, alpha)
            
            # Generate candidates for acquisition
            # Use random sampling for EI maximization
            n_candidates = 1000
            X_cand = np.random.uniform(-5, 5, (n_candidates, dim))
            
            # K(X_cand, X)
            K_s = matern52_kernel(X_cand, X, l, sigma_f)
            
            # Mean and Variance
            mean = K_s @ alpha
            v = np.linalg.solve(L, K_s.T)
            var = sigma_f**2 - np.sum(v**2, axis=1)
            std = np.sqrt(np.maximum(var, 1e-8))
            
            # Expected Improvement
            best_y = np.min(y)
            z = (best_y - mean) / std
            EI = (best_y - mean) * stats.norm.cdf(z) + std * stats.norm.pdf(z)
            
            # Select best candidate
            idx_best = np.argmax(EI)
            x_next = X_cand[idx_best]
            
            # Evaluate
            y_next = call_func(x_next)
            
            # Update data
            X = np.vstack([X, x_next])
            y = np.append(y, y_next)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
