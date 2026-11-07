"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """Bayesian Optimizer with Gaussian Process surrogate."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin for 300s limit
        
        # 1. Initialization with random sampling
        n_init = 15
        X_init = np.random.uniform(-5, 5, (n_init, self.dim))
        Y_init = np.array([func(x) for x in X_init])
        
        X = X_init
        Y = Y_init
        best_x = X_init[np.argmin(Y_init)]
        best_y = Y_init.min()
        calls = n_init
        
        # 2. Define Kernel
        # Matern 5/2 is good for non-smooth functions like Bent Cigar
        # ARD (Anisotropic) allows different length scales per dimension
        length_scale = np.ones(self.dim)
        # Initial length scales around 1.0
        kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, 
                                              length_scale_bounds=(0.1, 10.0), 
                                              nu=2.5)
        # WhiteKernel is for noise, but function is deterministic. 
        # We add a tiny white kernel to handle numerical stability if needed, 
        # but usually alpha in GPR handles this.
        
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, 
                                       n_restarts_optimizer=2, random_state=42)
        
        # Fit initial model
        try:
            gpr.fit(X, Y)
        except Exception:
            # If fitting fails, fall back to random search for the rest
            while calls < self.budget and time.time() < deadline:
                x_next = np.random.uniform(-5, 5, self.dim)
                y_next = func(x_next)
                X = np.vstack([X, [x_next]])
                Y = np.append(Y, y_next)
                calls += 1
                if y_next < best_y:
                    best_y = y_next
                    best_x = x_next
            return best_x, best_y

        # 3. Bayesian Optimization Loop
        # We will run BO for a portion of the budget, then switch to local refinement
        bo_budget = int(self.budget * 0.8) # 80% for BO
        
        while calls < bo_budget and time.time() < deadline:
            # Re-fit hyperparameters periodically to adapt to the landscape
            if calls % 20 == 0:
                try:
                    gpr.fit(X, Y)
                except Exception:
                    pass # Keep old model if refit fails

            # Define Expected Improvement acquisition function
            def neg_ei(x):
                mu, sigma = gpr.predict(x.reshape(1, -1), return_std=True)
                sigma = np.maximum(sigma, 1e-9)
                improvement = best_y - mu
                z = improvement / sigma
                ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei

            # Maximize EI by minimizing -EI
            # Use multiple random starts to find a good candidate
            best_x_acq = None
            best_ei_val = -np.inf
            
            # 5 random starts for the acquisition function optimization
            for _ in range(5):
                x0 = np.random.uniform(-5, 5, self.dim)
                res = minimize(neg_ei, x0, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6})
                if res.success:
                    # Clip to bounds
                    x_candidate = np.clip(res.x, -5, 5)
                    val = res.fun
                    if val < best_ei_val:
                        best_ei_val = val
                        best_x_acq = x_candidate
            
            if best_x_acq is None:
                best_x_acq = np.random.uniform(-5, 5, self.dim)
            else:
                best_x_acq = np.clip(best_x_acq, -5, 5)

            y_next = func(best_x_acq)
            X = np.vstack([X, [best_x_acq]])
            Y = np.append(Y, y_next)
            calls += 1
            
            if y_next < best_y:
                best_y = y_next
                best_x = best_x_acq

        # 4. Local Refinement Phase
        # Use Nelder-Mead to polish the best solution found
        if time.time() < deadline and calls < self.budget:
            # Start from the best point found
            x_local = best_x.copy()
            
            def local_func(x):
                nonlocal calls
                if time.time() > deadline or calls >= self.budget:
                    return 1e10 # Penalize if out of bounds/time
                x = np.clip(x, -5, 5)
                val = func(x)
                calls += 1
                if val < best_y:
                    best_y = val
                    best_x = x
                return val

            # Run Nelder-Mead
            try:
                res = minimize(local_func, x_local, method='Nelder-Mead', 
                               options={'maxiter': self.budget - calls, 'xatol': 1e-6, 'fatol': 1e-10})
            except Exception:
                pass

        return best_x, best_y
# EVOLVE-BLOCK-END
