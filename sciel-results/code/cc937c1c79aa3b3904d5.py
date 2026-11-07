import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 295.0
        n = self.dim
        B = self.budget

        best_x = np.zeros(n)
        best_f = float('inf')
        calls = 0

        def time_ok():
            return (time.time() - start) < deadline

        def ev(x):
            nonlocal calls, best_x, best_f
            if calls >= B:
                return float('inf')
            calls += 1
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Initial Sampling
        # Use a mix of random and structured points to get a good initial landscape
        n_init = min(40, B)
        # Start with a few random points
        for i in range(n_init):
            if not time_ok() or calls >= B:
                break
            x = rng.uniform(-5, 5, size=n)
            ev(x)

        X = [best_x.copy()]
        Y = [best_f]
        # Add a few more diverse points to the model if available
        # We'll just re-evaluate some points? No, we need new points.
        # Let's just use the best point and a few neighbors for the first model
        # Actually, for a robust start, let's do a few more random evals if we have budget
        for i in range(min(10, B - calls)):
            if not time_ok() or calls >= B:
                break
            x = rng.uniform(-5, 5, size=n)
            f = ev(x)
            X.append(x.copy())
            Y.append(f)
        
        X = np.array(X)
        Y = np.array(Y)

        # GP Parameters
        # Normalize Y
        y_mean = np.mean(Y)
        y_std = np.std(Y)
        if y_std < 1e-10:
            y_std = 1.0
        Y_norm = (Y - y_mean) / y_std

        length_scale = 1.0
        noise = 1e-8

        def kernel(X1, X2):
            sq_dists = np.sum(X1**2, axis=1)[:, None] - 2 * X1 @ X2.T + np.sum(X2**2, axis=1)[None, :]
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-0.5 * sq_dists / (length_scale**2))

        class GPM:
            def __init__(self, X, Y, ls, noise):
                self.X = X
                self.Y = Y
                self.ls = ls
                self.noise = noise
                self.K = kernel(X, X)
                self.K += noise * np.eye(len(X))
                try:
                    self.K_inv = np.linalg.inv(self.K)
                    self.alpha = self.K_inv @ Y
                except np.linalg.LinAlgError:
                    # Fallback to ridge regression if singular
                    lam = 1e-6
                    self.K_inv = np.linalg.inv(self.K + lam * np.eye(len(X)))
                    self.alpha = self.K_inv @ Y

            def predict(self, x_new):
                if x_new.ndim == 1:
                    x_new = x_new.reshape(1, -1)
                K_s = kernel(self.X, x_new) # (N, M)
                mu = K_s.T @ self.alpha
                v = K_s.T @ self.K_inv @ K_s
                var = 1.0 - np.diag(v)
                var = np.maximum(var, 0.0)
                std = np.sqrt(var)
                return mu, std

        def ei(x, gp_model, best_y_norm):
            mu, std = gp_model.predict(x)
            if std < 1e-10:
                return 0.0
            z = (best_y_norm - mu) / std
            ei_val = (best_y_norm - mu) * norm.cdf(z) + std * norm.pdf(z)
            return ei_val

        # Phase 2: Surrogate-Guided Search
        # We will use a mix of EI maximization and local polish
        
        # To speed up EI maximization, we can use a grid + local refinement
        # Or just multi-start Nelder-Mead on -EI
        
        surrogate_iterations = 0
        max_surrogate_iters = B - len(X)
        
        # Switch to local polish if we are close to the end or if EI is low
        local_polish_start = 0.7 * B
        
        while calls < B and time_ok():
            if len(X) >= local_polish_start:
                break
                
            # Fit GP
            X_arr = np.array(X)
            Y_arr = np.array(Y)
            y_mean = np.mean(Y_arr)
            y_std = np.std(Y_arr)
            if y_std < 1e-10:
                y_std = 1.0
            Y_norm = (Y_arr - y_mean) / y_std
            best_y_norm = (best_f - y_mean) / y_std
            
            gp_model = GPM(X_arr, Y_norm, length_scale, noise)
            
            # Maximize EI
            # Multi-start local search
            best_ei_val = -np.inf
            best_x_ei = None
            
            # Start points: random, best_x, center
            start_points = [
                best_x.copy(),
                np.zeros(n),
                rng.uniform(-5, 5, size=n),
                rng.uniform(-5, 5, size=n),
                rng.uniform(-5, 5, size=n)
            ]
            
            # Use a simple gradient-free optimization for EI
            # Since scipy.optimize.minimize is available, we can use Nelder-Mead
            from scipy.optimize import minimize
            
            for sp in start_points:
                if not time_ok():
                    break
                def neg_ei(x):
                    x = np.clip(x, -5, 5)
                    return -ei(x, gp_model, best_y_norm)
                
                try:
                    res = minimize(neg_ei, sp, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
                    if res.fun < best_ei_val:
                        best_ei_val = res.fun
                        best_x_ei = res.x
                except Exception:
                    continue
            
            if best_x_ei is None:
                break
                
            # Evaluate
            f_new = ev(best_x_ei)
            surrogate_iterations += 1
            
            # Update X, Y
            X.append(best_x_ei.copy())
            Y.append(f_new)
            
            # If no improvement for a while, maybe adjust strategy?
            # For now, just continue.
            
        # Phase 3: Local Polish
        if calls < B and time_ok():
            # Use Nelder-Mead or CMA-ES
            # Nelder-Mead is simple
            x0 = best_x.copy()
            
            # Simple Nelder-Mead implementation or use scipy
            def obj(x):
                if not time_ok() or calls >= B:
                    return float('inf')
                return ev(x)
            
            # Use scipy minimize
            from scipy.optimize import minimize
            try:
                res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            except Exception:
                pass
            
        return best_x, best_f
