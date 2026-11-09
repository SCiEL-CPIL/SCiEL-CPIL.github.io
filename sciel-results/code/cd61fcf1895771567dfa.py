"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

class Optimizer:
    """Bayesian Optimization with Gaussian Process surrogate."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        lb = -5.0
        ub = 5.0
        box = np.array([(lb, ub)] * dim)
        
        # Deadline guard: 330s limit, aim for 300s
        start_time = time.time()
        deadline = start_time + 290.0
        
        # Initial points: 10 random points to seed the GP
        n_init = 10
        X_init = np.random.uniform(lb, ub, size=(n_init, dim))
        Y_init = np.array([func(x) for x in X_init])
        
        X = X_init
        Y = Y_init
        best_x = X_init[np.argmin(Y_init)]
        best_f = np.min(Y_init)
        calls_made = n_init
        
        # Kernel: Matern 5/2 is robust for smooth but non-analytic functions like Schwefel
        # Length scales initialized to 1.0, variance to range of Y
        kernel = Matern(nu=2.5, length_scale=np.ones(dim), length_scale_bounds=(0.1, 10.0))
        
        # Acquisition function: Expected Improvement (EI)
        def acquisition(x):
            # x is 1D array of shape (dim,)
            x = np.array(x).reshape(1, -1)
            try:
                mu, sigma = gp.predict(x, return_std=True)
            except Exception:
                # If GP fails, return low value to avoid selection
                return -1e10
            
            sigma = np.maximum(sigma, 1e-8)
            # Improvement is defined as best_f - f(x)
            imp = best_f - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            # We want to maximize EI, so return -EI for minimization
            return -ei[0]
        
        # Optimization loop
        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Update GP model
            # Normalize Y for better GP conditioning? 
            # For simplicity and speed, we use raw Y, but Matern handles scale via hyperparams.
            # However, if Y range is large, GP might struggle. 
            # Let's standardize Y for the GP training, but predict in raw scale?
            # Actually, sklearn GP doesn't support target scaling directly in a simple way without manual work.
            # Given the small dimension (5) and budget (500), raw Y should be okay if we set kernel variance correctly.
            
            # Update kernel hyperparameters based on current best to handle scale changes?
            # A common trick: set the variance of the kernel to the variance of Y or range.
            # We'll re-initialize the kernel variance occasionally or just let it adapt.
            # For speed, we use a fixed kernel structure but allow length scales to adapt.
            
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=1e-10)
            
            # Train GP
            try:
                gp.fit(X, Y)
            except Exception:
                # If GP fitting fails, fallback to random search for the rest of the budget
                # This is a safety net
                while calls_made < budget:
                    if time.time() > deadline: break
                    x_rand = np.random.uniform(lb, ub, dim)
                    f_rand = func(x_rand)
                    if f_rand < best_f:
                        best_f = f_rand
                        best_x = x_rand
                    X = np.vstack([X, x_rand])
                    Y = np.append(Y, f_rand)
                    calls_made += 1
                break
            
            # Optimize the acquisition function
            # Use multi-start to handle multimodality of EI
            n_starts = 5
            best_acq_val = -np.inf
            best_x_next = None
            
            # Start points: random, and around current best
            starts = []
            for _ in range(n_starts - 1):
                starts.append(np.random.uniform(lb, ub, dim))
            starts.append(best_x)
            
            for x0 in starts:
                try:
                    res = minimize(acquisition, x0, method='L-BFGS-B', 
                                   bounds=box, options={'maxiter': 20})
                    if res.success and res.fun < best_acq_val:
                        best_acq_val = res.fun
                        best_x_next = res.x
                except Exception:
                    continue
            
            if best_x_next is None:
                # Fallback to random if optimization failed
                best_x_next = np.random.uniform(lb, ub, dim)
            
            # Evaluate the next point
            f_next = func(best_x_next)
            calls_made += 1
            
            # Update best
            if f_next < best_f:
                best_f = f_next
                best_x = best_x_next.copy()
            
            # Add to dataset
            X = np.vstack([X, best_x_next])
            Y = np.append(Y, f_next)
            
            # Periodic restart of kernel hyperparameters to avoid getting stuck in bad local optima of the GP itself?
            # Or just let it run.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
