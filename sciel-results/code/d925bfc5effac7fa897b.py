import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325 # 5s margin
        
        best_x = None
        best_f = np.inf
        calls = 0
        dim = self.dim
        budget = self.budget
        
        # Initial random sampling to seed the GP
        n_init = min(20, budget)
        X_init = np.random.uniform(-5, 5, size=(n_init, dim))
        y_init = np.empty(n_init)
        
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = X_init[i]
            x = np.clip(x, -5, 5)
            try:
                f = float(func(x))
            except:
                f = 1e9
            calls += 1
            y_init[i] = f
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if best_x is None:
             x0 = np.zeros(dim)
             try:
                 f = float(func(x0))
             except:
                 f = 1e9
             calls += 1
             best_f = f
             best_x = x0
             X_init = x0.reshape(1, -1)
             y_init = np.array([f])
             n_init = 1

        # Prepare data for GP
        X_obs = X_init[:n_init]
        y_obs = y_init[:n_init]
        
        # Kernel: Constant * RBF + White
        # RBF lengthscale is initialized to 1.0, will be optimized
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e1)) + WhiteKernel(1e-2, (1e-5, 1e1))
        
        # Candidate grid for EI optimization
        # A dense grid in 5D is expensive. We use a coarse grid and then local refinement?
        # Or just a moderate grid. 10 points per dim -> 100k points.
        # GP prediction on 100k points might be slow.
        # Let's try 5 points per dim -> 3125 points. This is fast.
        n_grid = 5
        grid_points = np.linspace(-5, 5, n_grid)
        grid = np.meshgrid(*([grid_points] * dim))
        grid = np.vstack([g.flatten() for g in grid])
        
        # Main BO loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Fit GP
            try:
                gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-8, normalize_y=True, n_restarts_optimizer=5)
                gp.fit(X_obs, y_obs)
            except Exception:
                # Fallback to random sampling if GP fails
                x_next = np.random.uniform(-5, 5, size=dim)
                try:
                    f_next = float(func(x_next))
                except:
                    f_next = 1e9
                calls += 1
                if f_next < best_f:
                    best_f = f_next
                    best_x = x_next.copy()
                X_obs = np.vstack([X_obs, x_next])
                y_obs = np.append(y_obs, f_next)
                continue

            # Calculate EI
            mu, sigma = gp.predict(grid, return_std=True)
            sigma = np.maximum(sigma, 1e-8) # Avoid division by zero
            
            # Expected Improvement
            improvement = best_f - mu
            z = improvement / sigma
            EI = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            
            # Select best candidate
            idx = np.argmax(EI)
            x_next = grid[idx]
            
            # Evaluate
            x_next = np.clip(x_next, -5, 5)
            try:
                f_next = float(func(x_next))
            except:
                f_next = 1e9
            calls += 1
            
            if f_next < best_f:
                best_f = f_next
                best_x = x_next.copy()
            
            # Update observations
            X_obs = np.vstack([X_obs, x_next])
            y_obs = np.append(y_obs, f_next)
            
            # Refine grid around best point? 
            # To save time, we stick to the fixed grid. 
            # If we have time, we could do a local search around best_x.
            # But GP should guide us.
            
            # Check time again before next iteration
            if time.time() > deadline:
                break
                
        return best_x, best_f
