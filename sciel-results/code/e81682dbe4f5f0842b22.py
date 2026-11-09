import numpy as np
import time
from scipy.optimize import minimize
import scipy.stats as stats

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        X_data = []
        Y_data = []

        def track_call(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                return best_f
            x = np.asarray(x, dtype=float).flatten()
            if x.shape[0] != dim:
                return best_f
            val = func(x)
            calls_used += 1
            if val < best_f:
                best_f = val
                best_x = np.array(x)
            X_data.append(x)
            Y_data.append(val)
            return val

        # Initialize with a few random points to establish the GP
        # Use Latin Hypercube Sampling for better initial coverage
        np.random.seed(42)
        initial_points = []
        # Simple LHS for dim=5
        for i in range(10):
            # Random point in [-5, 5]^dim
            p = np.random.uniform(-5, 5, dim)
            initial_points.append(p)
        
        # Evaluate initial points
        for p in initial_points:
            if time.time() > deadline - 5.0 or calls_used >= budget:
                break
            track_call(p)

        # GP Parameters
        # Matern kernel with nu=2.5 is a good default for smooth-ish functions
        # We use a simple RBF kernel for robustness and speed, as dim is small.
        # Actually, Matern 5/2 is often better for non-differentiable functions, but Schaffers is differentiable.
        # Let's use RBF (Squared Exponential) for stability.
        
        def kernel(X1, X2, length_scale, variance):
            # X1: (n1, d), X2: (n2, d)
            # Distance matrix
            # ||x - y||^2 = ||x||^2 - 2x.y + ||y||^2
            X1_sq = np.sum(X1**2, axis=1)[:, None]
            X2_sq = np.sum(X2**2, axis=1)[None, :]
            dist_sq = X1_sq - 2 * np.dot(X1, X2.T) + X2_sq
            dist_sq = np.maximum(dist_sq, 0)
            K = variance * np.exp(-0.5 * dist_sq / (length_scale**2))
            return K

        def predict_gp(X_new, length_scale, variance, noise):
            # X_new: (m, d)
            X_obs = np.array(X_data)
            Y_obs = np.array(Y_data)
            n = len(X_obs)
            if n == 0:
                return np.zeros(len(X_new)), np.zeros(len(X_new))
            
            K = kernel(X_obs, X_obs, length_scale, variance) + noise * np.eye(n)
            K_s = kernel(X_obs, X_new, length_scale, variance)
            
            # Solve K * alpha = Y
            try:
                alpha = np.linalg.solve(K, Y_obs)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, Y_obs, rcond=None)[0]
                
            # Mean
            mu = K_s.T @ alpha
            
            # Variance
            K_ss = kernel(X_new, X_new, length_scale, variance)
            var = np.diag(K_ss) - np.sum(K_s.T * (K @ K_s.T), axis=0)
            var = np.maximum(var, 1e-10)
            
            return mu, var

        # Adaptive Length Scale and Variance
        # Estimate initial length scale based on domain size
        ls = 1.5
        var = 100.0
        noise = 1e-6

        # Main Loop
        while calls_used < budget and time.time() < deadline - 5.0:
            # 1. Update GP parameters (simple heuristic)
            # Estimate variance from data
            if len(Y_data) > 1:
                var = np.var(Y_data) + 1e-6
            # Keep noise small
            
            # 2. Acquire next point using Expected Improvement (EI)
            # EI(x) = (f_min - mu(x)) * Phi(z) + sigma(x) * phi(z)
            # where z = (f_min - mu(x)) / sigma(x)
            
            # We need to maximize EI.
            # Since we can't sample infinitely, we use a grid + local optimization.
            
            # Candidate grid
            # Create a coarse grid
            grid_size = 5
            axes = np.linspace(-5, 5, grid_size)
            grid_points = np.array(np.meshgrid(*[axes]*dim)).T.reshape(-1, dim)
            
            # Evaluate EI on grid
            mu_grid, var_grid = predict_gp(grid_points, ls, var, noise)
            sigma_grid = np.sqrt(var_grid)
            
            f_min = best_f
            z_grid = (f_min - mu_grid) / (sigma_grid + 1e-10)
            ei_grid = (f_min - mu_grid) * stats.norm.cdf(z_grid) + sigma_grid * stats.norm.pdf(z_grid)
            
            # Find best grid point
            best_idx = np.argmax(ei_grid)
            x_start = grid_points[best_idx]
            
            # Refine with local optimization (Nelder-Mead on -EI)
            # We need to define a function that takes x, computes EI, and returns -EI
            def neg_ei(x):
                x = np.clip(x, -5, 5)
                mu, var = predict_gp(x.reshape(1, -1), ls, var, noise)
                sigma = np.sqrt(var[0])
                z = (f_min - mu[0]) / (sigma + 1e-10)
                ei = (f_min - mu[0]) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
                return -ei

            # Optimize
            try:
                res = minimize(neg_ei, x_start, method='Nelder-Mead', 
                               options={'xatol': 1e-3, 'fatol': 1e-3, 'maxiter': 50})
                x_next = res.x
            except Exception:
                x_next = x_start
                
            # Evaluate the selected point
            track_call(x_next)
            
            # Adaptive Length Scale Update (very simple)
            # If we improved significantly, we might want to zoom in (decrease ls)
            # If not, zoom out (increase ls)
            # This is heuristic.
            if len(Y_data) > 10:
                # Estimate local variance
                recent_vars = np.var(Y_data[-5:])
                if recent_vars < var * 0.1:
                    ls *= 0.9
                else:
                    ls *= 1.1
                ls = np.clip(ls, 0.5, 5.0)

        return best_x, best_f
