"""Bayesian Optimization optimizer for BBOB f14."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import minimize

class Optimizer:
    """Bayesian Optimization using Gaussian Process surrogate and Expected Improvement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0  # Safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        calls = 0

        # Hyperparameters for GP
        # Length scales for 5 dimensions
        # Use a kernel that allows for varying smoothness
        # We will implement a simple Matern 5/2 kernel with automatic relevance determination
        
        # Initialize with Latin Hypercube Sampling
        # 10 initial points is reasonable for 5D to start
        n_init = 10
        X_init = np.random.rand(n_init, dim) * 10 - 5  # Range [-5, 5]
        
        Y_init = np.zeros(n_init)
        best_x = None
        best_f = np.inf
        
        for i in range(n_init):
            if calls >= budget:
                break
            x = X_init[i]
            f = func(x)
            calls += 1
            Y_init[i] = f
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # If we ran out of budget during init, return best
        if calls >= budget:
            return best_x, best_f

        X_obs = X_init[:calls].copy()
        Y_obs = Y_init[:calls].copy()
        
        # Normalize Y to [0, 1] or standardize? 
        # GP is scale invariant if we normalize, but let's just use raw Y and let kernel handle it?
        # Better to standardize Y for numerical stability in GP fitting
        Y_mean = np.mean(Y_obs)
        Y_std = np.std(Y_obs)
        if Y_std < 1e-10:
            Y_std = 1.0
        Y_norm = (Y_obs - Y_mean) / Y_std

        # GP Hyperparameters: [length_scale_1, ..., length_scale_dim, signal_variance, noise_variance]
        # Initialize with reasonable values
        l0 = 2.0 * np.ones(dim)
        sigma_f0 = 1.0
        sigma_n0 = 1e-5
        
        # We will optimize hyperparameters every few iterations
        # For speed, we might fix hyperparameters or optimize them less frequently.
        # Given 500 budget, optimizing hyperparameters every 10-20 steps is feasible.
        
        def compute_kernel(X1, X2, hyperparams):
            # Matern 5/2 kernel
            l = hyperparams[:dim]
            sigma_f = hyperparams[dim]
            
            # Squared Euclidean distance
            sq_dist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dist = np.maximum(sq_dist, 0)
            
            # Matern 5/2: (1 + sqrt(3)*d + 3/5*d^2) * exp(-sqrt(3)*d) / (l^2)
            # Actually, standard form: K(d) = sigma_f^2 * (1 + sqrt(3)*d/l + 3/5*(d/l)^2) * exp(-sqrt(3)*d/l)
            # Let r = d/l
            r = np.sqrt(sq_dist) / l[None, :]
            
            K = sigma_f**2 * (1 + np.sqrt(3)*r + 3/5*r**2) * np.exp(-np.sqrt(3)*r)
            
            # Add noise on diagonal
            sigma_n = hyperparams[dim+1]
            if X1.shape[0] == X2.shape[0]:
                K[np.diag_indices_from(K)] += sigma_n**2
                
            return K

        def log_marginal_likelihood(hyperparams, X, Y):
            try:
                K = compute_kernel(X, X, hyperparams)
                # Add small jitter for numerical stability
                K += 1e-8 * np.eye(X.shape[0])
                
                # Cholesky decomposition
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    return -1e10
                    
                # Solve L alpha = Y
                alpha = np.linalg.solve(L, Y)
                
                # Log marginal likelihood: -0.5 * Y^T K^-1 Y - 0.5 * log(det(K)) - N/2 * log(2*pi)
                det_K = 2 * np.sum(np.log(np.diag(L)))
                log_mll = -0.5 * Y @ alpha - 0.5 * det_K - len(Y)/2 * np.log(2 * np.pi)
                
                return log_mll
            except Exception:
                return -1e10

        # Optimize hyperparameters initially
        # Use a local optimizer on the negative log marginal likelihood
        # Bounds for hyperparameters
        # Length scales: [0.1, 10]
        # Signal variance: [0.01, 10]
        # Noise variance: [1e-6, 1]
        
        bounds = [(0.1, 10.0)] * dim + [(0.01, 10.0), (1e-6, 1.0)]
        x0 = np.concatenate([l0, [sigma_f0, sigma_n0]])
        
        # Optimize hyperparameters
        res = minimize(lambda h: -log_marginal_likelihood(h, X_obs, Y_norm), 
                       x0, bounds=bounds, method='L-BFGS-B', options={'maxiter': 50})
        
        if res.success:
            opt_hyperparams = res.x
        else:
            opt_hyperparams = x0

        # Acquisition function: Expected Improvement (EI)
        def acquisition_function(x, X, Y, hyperparams, best_y, xi=0.01):
            # x is a single point
            x = x.reshape(1, -1)
            
            K_x = compute_kernel(x, X, hyperparams)
            K_xx = compute_kernel(x, x, hyperparams)
            
            # Add noise to K_xx
            # K_xx is 1x1
            K_x = K_x.flatten()
            
            # Covariance matrix
            K = compute_kernel(X, X, hyperparams)
            K += 1e-8 * np.eye(X.shape[0])
            
            try:
                # Mean and variance at x
                # mu_x = K_x^T K^-1 Y
                # var_x = K_xx - K_x^T K^-1 K_x
                K_inv_K_x = np.linalg.solve(K, K_x)
                mu = K_x @ K_inv_K_x @ Y
                var = K_xx[0,0] - K_x @ K_inv_K_x
                
                var = max(var, 1e-10)
                std = np.sqrt(var)
                
                # Normalize Y for EI calculation? 
                # EI is defined on the normalized scale
                # We want to minimize, so we maximize the improvement of the best value seen so far
                # best_y is the minimum Y_norm seen so far
                # Improvement = best_y - y_new
                # EI = E[max(0, best_y - y_new)]
                
                if std < 1e-10:
                    return 0.0
                    
                z = (best_y - mu) / std
                EI = (best_y - mu) * norm.cdf(z) + std * norm.pdf(z)
                
                # Add exploration term? No, EI handles it.
                # Penalize points close to observed points? EI does this naturally.
                
                return EI
            except Exception:
                return 0.0

        # Main loop
        # We will evaluate a batch of points? No, BO is sequential.
        # But we can parallelize the acquisition optimization? No, just sequential.
        
        # To speed up, we can use a local search around the best point to find the acquisition optimum.
        # Or use a grid? Grid is too expensive in 5D.
        # Let's use a few random restarts for the acquisition optimization.
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Update hyperparameters periodically
            if (calls - n_init) % 20 == 0:
                # Re-optimize hyperparameters
                Y_mean = np.mean(Y_obs)
                Y_std = np.std(Y_obs)
                if Y_std < 1e-10:
                    Y_std = 1.0
                Y_norm = (Y_obs - Y_mean) / Y_std
                
                res = minimize(lambda h: -log_marginal_likelihood(h, X_obs, Y_norm), 
                               opt_hyperparams, bounds=bounds, method='L-BFGS-B', options={'maxiter': 30})
                if res.success:
                    opt_hyperparams = res.x
                else:
                    # Keep old if failed
                    pass
                
                # Update best_y in normalized scale
                best_y_norm = (best_f - Y_mean) / Y_std
            
            # Find the next point to evaluate
            # Maximize acquisition function
            # Use multiple random starts
            best_acq = -np.inf
            best_x_next = None
            
            n_restarts = 5
            for _ in range(n_restarts):
                # Start near best point or random
                if np.random.rand() < 0.5:
                    x_start = best_x + 0.5 * np.random.randn(dim)
                else:
                    x_start = np.random.rand(dim) * 10 - 5
                
                # Clip to bounds
                x_start = np.clip(x_start, -5, 5)
                
                # Local optimization
                res = minimize(lambda x: -acquisition_function(x, X_obs, Y_norm, opt_hyperparams, best_y_norm), 
                               x_start, bounds=[(-5, 5)] * dim, method='L-BFGS-B', options={'maxiter': 50})
                
                if res.success:
                    acq_val = -res.fun
                    if acq_val > best_acq:
                        best_acq = acq_val
                        best_x_next = res.x
            
            if best_x_next is None:
                # Fallback to random
                best_x_next = np.random.rand(dim) * 10 - 5
            
            best_x_next = np.clip(best_x_next, -5, 5)
            
            # Evaluate
            f_new = func(best_x_next)
            calls += 1
            
            # Update observed data
            X_obs = np.vstack([X_obs, best_x_next])
            Y_obs = np.vstack([Y_obs, f_new])
            
            # Update best
            if f_new < best_f:
                best_f = f_new
                best_x = best_x_next.copy()
                
        return best_x, best_f

# EVOLVE-BLOCK-END
