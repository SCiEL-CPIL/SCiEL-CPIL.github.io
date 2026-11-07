import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290
        
        call_count = 0
        def wrapped_func(x):
            nonlocal call_count
            if call_count >= self.budget:
                raise RuntimeError("Budget exceeded")
            call_count += 1
            return func(x)

        rng = np.random.default_rng(42)
        dim = self.dim
        
        # Phase 1: Bayesian Optimization with GP surrogate
        # Initialize with a space-filling sample
        init_n = 20
        X_init = rng.uniform(-5, 5, (init_n, dim))
        y_init = np.array([wrapped_func(x) for x in X_init])
        
        X_obs = X_init
        y_obs = y_init
        
        best_x = X_obs[np.argmin(y_obs)]
        best_f = np.min(y_obs)
        
        # GP hyperparameters (fixed for simplicity and speed)
        length_scale = 1.5
        signal_var = 1.0
        noise_var = 1e-6
        
        def matern52_kernel(X1, X2, length_scale):
            # Compute pairwise distances
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=-1))
            return signal_var * np.sqrt(1 + dist/length_scale + (dist/length_scale)**2) * np.exp(-dist/length_scale)
        
        def gp_predict(X_new):
            K = matern52_kernel(X_obs, X_obs, length_scale) + noise_var * np.eye(len(X_obs))
            K_s = matern52_kernel(X_obs, X_new, length_scale)
            try:
                K_inv_y = np.linalg.solve(K, y_obs)
                K_inv_s = np.linalg.solve(K, K_s)
            except np.linalg.LinAlgError:
                # Fallback if singular
                K_inv_y = np.linalg.lstsq(K, y_obs, rcond=None)[0]
                K_inv_s = np.linalg.lstsq(K, K_s, rcond=None)[0]
            
            mu = K_s.T @ K_inv_y
            var = signal_var - np.sum(K_s * K_inv_s, axis=0)
            var = np.maximum(var, 1e-10)
            return mu, np.sqrt(var)
        
        def expected_improvement(X_new, best_f, mu, sigma):
            # EI for minimization
            # Z = (best_f - mu) / sigma
            Z = (best_f - mu) / (sigma + 1e-10)
            ei = (best_f - mu) * norm.cdf(Z) - sigma * norm.pdf(Z)
            # Handle case where sigma is very small
            ei = np.where(sigma > 1e-10, ei, 0)
            return ei
        
        bo_budget = 150
        for i in range(bo_budget):
            if time.time() > deadline - 60:
                break
            if call_count >= self.budget:
                break
            
            # Generate candidate points
            # Mix of random and local perturbations of best
            n_candidates = 100
            X_rand = rng.uniform(-5, 5, (n_candidates, dim))
            
            # Local perturbations around best few
            n_local = 50
            X_local = []
            for _ in range(n_local):
                idx = np.argmin(y_obs)
                center = X_obs[idx]
                step = 0.5 * rng.standard_normal(dim)
                child = np.clip(center + step, -5, 5)
                X_local.append(child)
            X_local = np.array(X_local)
            
            X_cand = np.vstack([X_rand, X_local])
            
            # Predict
            mu, sigma = gp_predict(X_cand)
            ei = expected_improvement(X_cand, best_f, mu, sigma)
            
            # Select best EI
            idx_best_ei = np.argmax(ei)
            x_new = X_cand[idx_best_ei]
            
            # Evaluate
            y_new = wrapped_func(x_new)
            
            # Update
            X_obs = np.vstack([X_obs, [x_new]])
            y_obs = np.append(y_obs, y_new)
            
            if y_new < best_f:
                best_f = y_new
                best_x = x_new
        
        # Phase 2: Local refinement with L-BFGS-B from best BO point
        remaining_budget = self.budget - call_count
        if remaining_budget > 10:
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': remaining_budget // 2, 'maxfun': remaining_budget}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
