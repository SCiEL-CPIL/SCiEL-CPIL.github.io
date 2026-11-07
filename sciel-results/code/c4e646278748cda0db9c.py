"""Bayesian Optimization with Gaussian Process for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.utils import check_array

class Optimizer:
    """
    Optimizer for BBOB f21 (Gallagher 101 peaks).
    Strategy:
    1. Global Exploration: Use a Latin Hypercube sample to initialize the surrogate.
    2. Bayesian Optimization: Use a Gaussian Process surrogate and an Expected Improvement
       acquisition function to iteratively select the next point.
    3. Exploit/Explore Balance: The acquisition function naturally balances exploration
       and exploitation, which is crucial for multimodal functions like f21.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # State
        best_x = None
        best_f = np.inf
        calls_made = 0
        X_obs = []  # List of observed points
        y_obs = []  # List of observed values

        def time_left():
            return deadline - time.time()

        def make_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or time_left() < 0.1:
                return None
            # Ensure x is a valid numpy array within bounds
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            
            try:
                f = func(x)
            except Exception:
                return np.inf
            
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: Global Initialization ---
        # Use Latin Hypercube Sampling for a well-spread initial set.
        n_init = 20
        # Simple LHS: divide each dimension into n_init intervals and sample one point per interval
        X_init = np.zeros((n_init, dim))
        for d in range(dim):
            perm = rng.permutation(n_init)
            for i in range(n_init):
                # Sample uniformly within the i-th interval of dimension d
                X_init[perm[i], d] = (i + rng.random()) / n_init * 10.0 - 5.0
        
        # Evaluate initial points
        for i in range(n_init):
            f = make_call(X_init[i])
            if f is None:
                break
            X_obs.append(X_init[i])
            y_obs.append(f)
        
        if not X_obs:
            return best_x, best_f

        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)

        # --- Phase 2: Bayesian Optimization Loop ---
        # Define the Gaussian Process model
        # Matern kernel with nu=2.5 is a good default for smooth-ish functions
        # Add a white kernel to handle noise (even if the function is deterministic)
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-10)
        
        # We will retrain the GP every few iterations to save time, or use a fixed model.
        # For simplicity and speed, let's retrain every 5 iterations.
        
        # Acquisition function: Expected Improvement (EI)
        def expected_improvement(X, model, y_best, xi=0.01):
            """
            Compute the Expected Improvement for a set of points X.
            X: (n_samples, dim)
            model: fitted GaussianProcessRegressor
            y_best: current best value
            xi: exploration-exploitation trade-off parameter
            """
            X = np.atleast_2d(X)
            if X.shape[0] == 1:
                X = X[0:1]
            
            try:
                mu, sigma = model.predict(X, return_std=True)
            except Exception:
                return np.zeros(X.shape[0])
            
            sigma = np.maximum(sigma, 1e-10)
            
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                Z = (y_best - mu - xi) / sigma
                EI = (y_best - mu - xi) * (1 - norm.cdf(Z)) + sigma * norm.pdf(Z)
                # Handle cases where sigma is very small
                EI = np.where(sigma < 1e-10, 0.0, EI)
            
            return EI

        # Import norm for CDF/PDF
        from scipy.stats import norm

        # Main BO loop
        max_bo_iters = 100  # Limit number of BO iterations to avoid excessive GP fitting time
        
        for bo_iter in range(max_bo_iters):
            if time_left() < 1.0 or calls_made >= budget:
                break
            
            # Retrain the GP model periodically or on every iteration if fast enough
            # For 5D and <200 points, GP fitting is fast. Let's retrain every 5 steps.
            if bo_iter % 5 == 0 or bo_iter == 0:
                try:
                    # Normalize y for better GP performance
                    y_mean = np.mean(y_obs)
                    y_std = np.std(y_obs) + 1e-10
                    y_scaled = (y_obs - y_mean) / y_std
                    
                    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=1)
                    gp.fit(X_obs, y_scaled)
                except Exception:
                    # If GP fails, fall back to random search
                    gp = None
            
            if gp is None:
                # Fallback: random sampling
                x_new = rng.uniform(-5.0, 5.0, size=dim)
            else:
                # Candidate generation: use a grid or random points
                # For 5D, a grid is expensive. Let's use random candidates.
                n_candidates = 100
                X_cand = rng.uniform(-5.0, 5.0, size=(n_candidates, dim))
                
                # Compute EI for candidates
                y_best_scaled = (best_f - y_mean) / y_std
                EI = expected_improvement(X_cand, gp, y_best_scaled, xi=0.01)
                
                # Select the candidate with the highest EI
                idx_best = np.argmax(EI)
                x_new = X_cand[idx_best]
            
            # Evaluate the new point
            f_new = make_call(x_new)
            if f_new is None:
                break
            
            # Update observations
            X_obs = np.vstack([X_obs, x_new])
            y_obs = np.append(y_obs, f_new)
            
            # Occasionally, if we are stuck, we can do a local refinement around the best point
            # But BO should handle this. Let's add a simple local search every 20 iterations.
            if bo_iter > 0 and bo_iter % 20 == 0 and best_x is not None:
                # Do a small Nelder-Mead step around the best point
                try:
                    res = minimize(func, best_x, method='Nelder-Mead', options={'maxiter': 20, 'xatol': 1e-4, 'fatol': 1e-6})
                    if res.fun < best_f:
                        best_x = res.x
                        best_f = res.fun
                        # Add this point to observations
                        X_obs = np.vstack([X_obs, best_x])
                        y_obs = np.append(y_obs, best_f)
                except Exception:
                    pass

        return best_x, best_f

# EVOLVE-BLOCK-END
