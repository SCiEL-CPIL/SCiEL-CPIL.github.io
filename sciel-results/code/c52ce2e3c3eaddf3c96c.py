"""Optimized program using Gaussian Process Bayesian Optimization."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """Bayesian Optimizer with Gaussian Process surrogate and Expected Improvement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        X = []  # List of points (dim,)
        Y = []  # List of values (float)
        
        x_best = None
        f_best = np.inf
        calls_made = 0

        def update_best(x, f):
            nonlocal x_best, f_best
            if f < f_best:
                f_best = f
                x_best = x

        # Phase 1: Initial random sampling to build a baseline
        n_init = 10
        for _ in range(n_init):
            if calls_made >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls_made += 1
            X.append(x)
            Y.append(f)
            update_best(x, f)

        if calls_made >= budget or time.time() > deadline:
            return x_best, f_best

        # GP Parameters
        # Fixed hyperparameters for simplicity and speed, tuned for [-5,5]^5
        # Length scale 1.0 allows local variations, variance 1.0
        length_scale = 1.0
        noise = 1e-6
        variance = 1.0

        def kernel(X1, X2, ls, var):
            # RBF Kernel
            sq_dists = np.sum(X1**2, 1)[:, None] - 2 * X1 @ X2.T + np.sum(X2**2, 1)[None, :]
            sq_dists = np.maximum(sq_dists, 0)
            return var * np.exp(-0.5 * sq_dists / (ls * ls))

        def predict_gp(X_train, Y_train, X_new, ls, var, noise):
            K = kernel(X_train, X_train, ls, var) + noise * np.eye(len(X_train))
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            
            K_s = kernel(X_new, X_train, ls, var)
            mu = K_s @ alpha
            v = np.linalg.solve(L, K_s.T)
            var_pred = var - np.sum(v**2, axis=1)
            var_pred = np.maximum(var_pred, 0)
            return mu, np.sqrt(var_pred)

        def expected_improvement(x, f_best, mu, sigma, xi=0.01):
            if sigma < 1e-8:
                return 0.0
            z = (f_best - mu - xi) / sigma
            ei = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        def acquisition_function(x, f_best_val, X_train, Y_train, ls, var, noise):
            # x is a single point (dim,)
            x_new = x.reshape(1, -1)
            mu, sigma = predict_gp(X_train, Y_train, x_new, ls, var, noise)
            return -expected_improvement(x, f_best_val, mu[0], sigma[0], xi=0.01)

        # Phase 2: GP-Based Search
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Check if we have enough data to fit GP
            if len(X) < 2:
                x = rng.uniform(-5, 5, dim)
                f = func(x)
                calls_made += 1
                X.append(x)
                Y.append(f)
                update_best(x, f)
                continue

            X_arr = np.array(X)
            Y_arr = np.array(Y)

            # Find next point using EI
            # We try multiple starting points for the optimizer to avoid local minima in EI
            # Starting points: current best, random points, and points around best
            start_points = [x_best]
            for _ in range(5):
                start_points.append(rng.uniform(-5, 5, dim))
            # Add points around best
            for _ in range(5):
                start_points.append(np.clip(x_best + rng.normal(0, 0.5, dim), -5, 5))

            best_x_next = None
            best_ei_val = np.inf

            for sp in start_points:
                sp = np.clip(sp, -5, 5)
                try:
                    res = minimize(
                        acquisition_function, 
                        sp, 
                        args=(f_best, X_arr, Y_arr, length_scale, variance, noise),
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50, 'ftol': 1e-6}
                    )
                    if res.success:
                        if res.fun < best_ei_val:
                            best_ei_val = res.fun
                            best_x_next = res.x
                except Exception:
                    continue

            if best_x_next is None:
                # Fallback to random if optimization failed
                best_x_next = rng.uniform(-5, 5, dim)

            # Evaluate the next point
            x_next = np.clip(best_x_next, -5, 5)
            f_next = func(x_next)
            calls_made += 1
            X.append(x_next)
            Y.append(f_next)
            update_best(x_next, f_next)

            # If we are very close to the minimum, we might want to switch to a local polish
            # but the GP should handle it. Let's add a simple local polish at the end.

        # Phase 3: Final Polish (Nelder-Mead style or simple gradient-free local search)
        if calls_made < budget and x_best is not None:
            remaining = budget - calls_made
            # Use a simple local search to refine x_best
            # This is safer than relying on GP if the GP model is slightly off near the minimum
            x_curr = x_best.copy()
            f_curr = f_best
            step_size = 0.1
            
            for _ in range(min(remaining, 50)):
                if time.time() > deadline or calls_made >= budget:
                    break
                
                # Try random directions
                improved = False
                for _ in range(dim * 2): # 10 tries per iteration
                    direction = rng.standard_normal(dim)
                    direction /= np.linalg.norm(direction)
                    x_trial = np.clip(x_curr + step_size * direction, -5, 5)
                    f_trial = func(x_trial)
                    calls_made += 1
                    
                    if f_trial < f_curr:
                        x_curr = x_trial
                        f_curr = f_trial
                        update_best(x_curr, f_curr)
                        improved = True
                        break
                    else:
                        step_size *= 0.9 # Shrink step if no improvement in this direction attempt? 
                        # Actually, standard Nelder-Mead logic is better, but simple random direction search works
                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-4:
                        break

        return x_best, f_best
