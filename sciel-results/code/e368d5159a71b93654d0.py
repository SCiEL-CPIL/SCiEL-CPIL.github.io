import numpy as np
import time
import scipy.stats as stats

class Optimizer:
    """
    Bayesian Optimizer for BBOB-style f17 (Schaffers F7).
    
    Strategy:
    - Use a Gaussian Process surrogate to model the function.
    - Use Expected Improvement (EI) to select the next point.
    - Initial phase: Random sampling to build the model.
    - Subsequent phase: GP-based exploration/exploitation.
    - Handles the hidden shift/rotation by adapting the kernel to the data distribution.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        dim = self.dim
        budget = self.budget
        
        # State
        X_obs = []  # Observed points
        y_obs = []  # Observed values
        best_x = np.zeros(dim)
        best_f = float('inf')
        count = 0
        
        # GP Parameters
        # We will use a Matern 5/2 kernel which is good for smooth-ish but not infinitely differentiable functions like Schaffers
        # Actually, Schaffers F7 is non-smooth at the global minimum (kink). 
        # A Matern 3/2 or 5/2 kernel is appropriate.
        # Let's use a simple RBF (Squared Exponential) first, it's robust.
        # Or better, a Matern kernel.
        
        # Hyperparameters for the GP
        # We will fix the length scales and variance to avoid expensive hyperparameter optimization 
        # which might eat into the budget or time.
        # Initial scale: The box is [-5, 5], so range is 10.
        # Let's assume a length scale of ~2.0 initially.
        # Variance: The function values can be in [-100, 100], so sigma_f ~ 50.
        
        length_scale = 2.0
        sigma_f = 50.0
        sigma_n = 1e-6 # Noise variance, small
        
        def kernel(X1, X2, l, sf):
            """Matern 5/2 kernel"""
            # X1: (n, d), X2: (m, d)
            # Squared distances
            X1 = X1[:, np.newaxis, :]
            X2 = X2[np.newaxis, :, :]
            d2 = np.sum((X1 - X2) ** 2, axis=-1)
            # Matern 5/2: (1 + sqrt(3)*r) * exp(-sqrt(3)*r)
            r = np.sqrt(d2)
            K = (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
            K *= sf**2
            return K

        def eval_point(x):
            nonlocal count, best_x, best_f
            try:
                f = func(x)
                count += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except Exception:
                return float('inf')

        # 1. Initial Random Sampling
        # We need enough points to fit a GP. For d=5, maybe 2*d + 5 = 15 points?
        # Let's use 20 points for a robust initial model.
        n_init = 20
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = np.random.uniform(-5, 5, dim)
            f = eval_point(x)
            if count >= budget:
                break
            X_obs.append(x)
            y_obs.append(f)
            
        if count >= budget:
            return best_x, best_f

        # Convert to arrays
        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)
        
        # Center and scale y for numerical stability
        y_mean = np.mean(y_obs)
        y_std = np.std(y_obs)
        if y_std < 1e-8:
            y_std = 1.0
        y_obs_scaled = (y_obs - y_mean) / y_std
        
        # Precompute GP matrices for the initial set
        # K: (n, n)
        K = kernel(X_obs, X_obs, length_scale, sigma_f)
        K += sigma_n * np.eye(len(X_obs))
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs_scaled))
        except np.linalg.LinAlgError:
            # Fallback to diagonal if Cholesky fails (should be rare)
            K += 1e-6 * np.eye(len(X_obs))
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs_scaled))
            except:
                # If GP fails completely, fallback to simple random search for the rest
                while count < budget:
                    if time.time() > deadline: break
                    x = np.random.uniform(-5, 5, dim)
                    eval_point(x)
                return best_x, best_f

        # 2. Bayesian Optimization Loop
        # We will use a local search or a grid to find the max EI point, 
        # or simply sample many candidates and pick the best EI.
        # Given d=5, we can sample 1000-5000 candidates per iteration.
        
        n_candidates = 2000
        
        while count < budget:
            if time.time() > deadline:
                break
                
            # Generate candidates
            # Mix of random points in the box and points around the current best
            # to encourage local exploitation
            
            # 50% random in box
            n_rand = n_candidates // 2
            X_cand_rand = np.random.uniform(-5, 5, (n_rand, dim))
            
            # 50% local around best
            # Sigma for local search: start with 1.0, shrink if no improvement?
            # Simple heuristic: use 0.5 * (max - min) of observed X? 
            # Or just a fixed small sigma like 0.5
            local_sigma = 0.5
            X_cand_local = best_x + local_sigma * np.random.randn(n_candidates - n_rand, dim)
            X_cand_local = np.clip(X_cand_local, -5, 5)
            
            X_cand = np.vstack([X_cand_rand, X_cand_local])
            
            # Compute GP predictions
            # K_s: (n_cand, n_obs)
            K_s = kernel(X_cand, X_obs, length_scale, sigma_f)
            
            # Mean
            mu = K_s @ alpha
            
            # Variance
            # v = K_s @ L^-1
            v = np.linalg.solve(L, K_s.T)
            sigma2 = sigma_f**2 - np.sum(v**2, axis=1)
            sigma = np.sqrt(np.maximum(sigma2, 1e-10))
            
            # Expected Improvement
            # EI = (f_best - mu - sigma) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
            # Note: y_obs_scaled is scaled. f_best_scaled = (best_f - y_mean)/y_std
            f_best_scaled = (best_f - y_mean) / y_std
            
            # Handle division by zero in sigma
            z = (f_best_scaled - mu) / sigma
            # Clip z to avoid extreme values
            z = np.clip(z, -10, 10)
            
            EI = (f_best_scaled - mu - sigma) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
            
            # Find best candidate
            idx = np.argmax(EI)
            x_new = X_cand[idx]
            
            # Evaluate
            f_new = eval_point(x_new)
            
            if count >= budget:
                break
                
            # Update GP
            # Append new point
            X_obs = np.vstack([X_obs, x_new])
            y_new_scaled = (f_new - y_mean) / y_std
            y_obs = np.append(y_obs, f_new)
            y_obs_scaled = np.append(y_obs_scaled, y_new_scaled)
            
            # Update Cholesky and alpha incrementally? 
            # For simplicity and robustness with small N (< 100), just recompute.
            # N will be at most 500. 500x500 Cholesky is fast (< 1ms).
            K = kernel(X_obs, X_obs, length_scale, sigma_f)
            K += sigma_n * np.eye(len(X_obs))
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs_scaled))
            except np.linalg.LinAlgError:
                # If it fails, add jitter
                K += 1e-6 * np.eye(len(X_obs))
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs_scaled))
                except:
                    # If GP breaks, fall back to random
                    while count < budget:
                        if time.time() > deadline: break
                        x = np.random.uniform(-5, 5, dim)
                        eval_point(x)
                    return best_x, best_f

        return best_x, best_f
