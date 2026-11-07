import numpy as np
import time
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.gaussian_process.kernels import RBF

class Optimizer:
    """Hybrid Bayesian Optimization + L-BFGS-B optimizer for BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Safety margin: 330s limit, stop at 320s
        deadline = time.time() + 320
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        bounds = [(-5, 5)] * dim
        lower = np.full(dim, -5.0)
        upper = np.full(dim, 5.0)
        
        def safe_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), lower, upper)
            if len(x) != dim:
                x = x[:dim]
            try:
                fx = float(func(x))
            except Exception:
                return np.inf
            calls += 1
            if np.isnan(fx) or np.isinf(fx):
                return np.inf
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial random sampling to seed the GP
        n_init = 15
        X_obs = []
        y_obs = []
        
        for i in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            if i == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-5, 5, dim)
            fx = safe_func(x)
            if not np.isinf(fx):
                X_obs.append(x)
                y_obs.append(fx)
        
        if not X_obs:
            # Fallback: just return what we have
            if best_x is None:
                best_x = np.zeros(dim)
                safe_func(best_x)
            return best_x, best_f
        
        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)
        
        # Phase 2: Bayesian Optimization loop
        # Use a significant portion of budget for BO, then refine with L-BFGS-B
        bo_budget = int(budget * 0.6)  # 60% for BO
        bo_calls_made = calls
        
        # GP setup
        # Matern kernel with nu=2.5 is a good default for black-box optimization
        # Length scales are initialized to 1.0, variance to 1.0
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(noise_level=1e-6)
        
        # We'll use a simple EI acquisition function
        # Since we don't have a dedicated BO library, we implement a simple version
        # by sampling candidate points and selecting the one with max EI
        
        # To keep it efficient, we'll sample a fixed number of candidates each iteration
        n_candidates = 50
        
        while calls < bo_budget and time.time() < deadline:
            if len(X_obs) < 2:
                # Not enough points for GP, just do random
                x = rng.uniform(-5, 5, dim)
                fx = safe_func(x)
                if not np.isinf(fx):
                    X_obs = np.vstack([X_obs, x])
                    y_obs = np.append(y_obs, fx)
                continue
            
            try:
                # Fit GP
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
                gp.fit(X_obs, y_obs)
                
                # Generate candidate points
                # Mix of random points and points near current best
                X_cand = []
                n_random_cand = int(n_candidates * 0.5)
                n_local_cand = n_candidates - n_random_cand
                
                # Random candidates
                X_cand.append(rng.uniform(-5, 5, (n_random_cand, dim)))
                
                # Local candidates around best_x
                if best_x is not None:
                    local_scale = 1.0  # Scale of perturbation
                    X_cand.append(np.clip(best_x + rng.normal(0, local_scale, (n_local_cand, dim)), lower, upper))
                else:
                    X_cand.append(rng.uniform(-5, 5, (n_local_cand, dim)))
                
                X_cand = np.vstack(X_cand)
                
                # Compute mean and std for candidates
                mu, sigma = gp.predict(X_cand, return_std=True)
                
                # Expected Improvement
                # EI = (y_best - mu) * Phi((y_best - mu) / sigma) + sigma * phi((y_best - mu) / sigma)
                # where y_best is the current best (minimum)
                # Since we are minimizing, y_best = min(y_obs)
                y_best = np.min(y_obs)
                
                # Avoid division by zero
                sigma = np.maximum(sigma, 1e-10)
                
                # Standard normal CDF and PDF
                from scipy.stats import norm
                z = (y_best - mu) / sigma
                ei = (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                
                # Select candidate with max EI
                idx = np.argmax(ei)
                x_next = X_cand[idx]
                
                # Evaluate
                fx = safe_func(x_next)
                
                if not np.isinf(fx):
                    X_obs = np.vstack([X_obs, x_next])
                    y_obs = np.append(y_obs, fx)
                    
            except Exception:
                # If GP fails, fall back to random
                x = rng.uniform(-5, 5, dim)
                fx = safe_func(x)
                if not np.isinf(fx):
                    X_obs = np.vstack([X_obs, x])
                    y_obs = np.append(y_obs, fx)
        
        # Phase 3: Local refinement with L-BFGS-B from best points
        # Use the top few points from BO as starting points
        if calls < budget and time.time() < deadline and best_x is not None:
            # Sort observed points by fitness
            sorted_indices = np.argsort(y_obs)
            top_k = min(5, len(sorted_indices))
            local_starts = X_obs[sorted_indices[:top_k]]
            
            # Also include the current best
            if best_x not in local_starts:
                local_starts = np.vstack([local_starts, best_x])
            
            max_fun_per_local = max(10, (budget - calls) // max(1, len(local_starts)))
            
            for start in local_starts:
                if calls >= budget or time.time() > deadline:
                    break
                
                try:
                    result = minimize(
                        safe_func,
                        start,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 30, 'maxfun': max(5, min(30, budget - calls - 5))}
                    )
                    
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass
            
            # Final polish
            if calls < budget and time.time() < deadline and best_x is not None:
                try:
                    result = minimize(
                        safe_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 50, 'maxfun': max(5, budget - calls)}
                    )
                    
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass
        
        return best_x, best_f
