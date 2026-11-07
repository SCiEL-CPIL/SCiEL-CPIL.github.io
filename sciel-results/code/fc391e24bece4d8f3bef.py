import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Storage for GP
        X_list = []
        Y_list = []
        
        # Phase 1: Bayesian Optimization
        # Phase 2: Local Search Polish
        
        # Hyperparameters for GP
        # Matern 5/2 kernel
        # l: lengthscales, sigma: signal variance
        # We standardize Y, so sigma is effectively 1 in the kernel calculation relative to std(Y)
        # But we need to map back.
        
        def matern52_kernel(X1, X2, lengthscales):
            # X1: (n, d), X2: (m, d)
            # lengthscales: (d,)
            # Compute squared distances
            X1_scaled = X1 / lengthscales
            X2_scaled = X2 / lengthscales
            sq_dists = np.sum(X1_scaled**2, axis=1)[:, None] + np.sum(X2_scaled**2, axis=1)[None, :] - 2 * X1_scaled @ X2_scaled.T
            sq_dists = np.maximum(sq_dists, 0)
            d = np.sqrt(sq_dists)
            sqrt5_d = np.sqrt(5) * d
            K = (1 + sqrt5_d + 5/3 * d**2) * np.exp(-sqrt5_d)
            return K

        def fit_gp(X, Y, lengthscales):
            n = len(Y)
            Y_std = (Y - np.mean(Y)) / (np.std(Y) + 1e-8)
            K = matern52_kernel(X, X, lengthscales)
            K += 1e-6 * np.eye(n) # Noise for stability
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_std))
            return L, alpha, np.mean(Y), np.std(Y) + 1e-8

        def predict_gp(x_new, L, alpha, y_mean, y_std, lengthscales):
            # x_new: (d,)
            x_new = x_new.reshape(1, -1)
            K_s = matern52_kernel(x_new, X, lengthscales) # (1, n)
            # K_s is (1, n)
            v = np.linalg.solve(L, K_s.T) # (n, 1)
            mu_std = (v.T @ alpha).item() # (1,) -> scalar
            mu = mu_std * y_std + y_mean
            
            # Variance
            # Var = 1 - K_s K^{-1} K_s^T
            # v is L^{-1} K_s^T. So K_s K^{-1} K_s^T = v^T v
            var_std = 1.0 - np.sum(v**2)
            var = var_std * y_std**2
            if var < 0: var = 0
            return mu, np.sqrt(var)

        def expected_improvement(x, best_y, L, alpha, y_mean, y_std, lengthscales):
            mu, sigma = predict_gp(x, L, alpha, y_mean, y_std, lengthscales)
            if sigma < 1e-8:
                return 0.0
            z = (best_y - mu) / sigma
            # EI for minimization
            # If mu < best_y, z > 0.
            # EI = (best_y - mu) * Phi(z) + sigma * phi(z)
            ei = (best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei # We want to maximize EI, so minimize -EI

        try:
            # 1. Initial Sampling
            n_init = 20
            for i in range(n_init):
                if calls_made >= budget or time.time() >= deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                calls_made += 1
                X_list.append(x)
                Y_list.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            X = np.array(X_list)
            Y = np.array(Y_list)
            
            # Initial lengthscales
            lengthscales = np.ones(dim) * 1.5
            
            # 2. BO Loop
            bo_budget = int(budget * 0.8) # Use 80% for BO
            polish_budget = budget - bo_budget
            
            while calls_made < bo_budget and time.time() < deadline:
                # Fit GP
                L, alpha, y_mean, y_std = fit_gp(X, Y, lengthscales)
                
                # Find max EI
                best_ei_val = -np.inf
                best_candidate = None
                
                # Multi-start optimization of EI
                n_starts = 10
                for _ in range(n_starts):
                    # Random start
                    x0 = rng.uniform(-5, 5, dim)
                    res = minimize(
                        expected_improvement, 
                        x0, 
                        args=(best_f, L, alpha, y_mean, y_std, lengthscales),
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50, 'ftol': 1e-6}
                    )
                    if res.fun < best_ei_val:
                        best_ei_val = res.fun
                        best_candidate = res.x
                
                if best_candidate is None:
                    break
                    
                # Evaluate best candidate
                fx = func(best_candidate)
                calls_made += 1
                X_list.append(best_candidate)
                Y_list.append(fx)
                X = np.array(X_list)
                Y = np.array(Y_list)
                
                if fx < best_f:
                    best_f = fx
                    best_x = best_candidate.copy()
                    
                # Optional: Update lengthscales if time permits? 
                # For simplicity and speed, keep fixed.
                
            # 3. Local Search Polish
            if calls_made < budget and time.time() < deadline and best_x is not None:
                x_curr = best_x.copy()
                f_curr = best_f
                step_size = 0.5
                no_improve = 0
                max_no_improve = 20
                
                while calls_made < budget and time.time() < deadline:
                    # Generate neighbor
                    direction = rng.standard_normal(dim)
                    x_try = x_curr + step_size * direction
                    x_try = np.clip(x_try, -5, 5)
                    
                    f_try = func(x_try)
                    calls_made += 1
                    
                    if f_try < f_curr:
                        x_curr = x_try
                        f_curr = f_try
                        if f_curr < best_f:
                            best_f = f_curr
                            best_x = x_curr.copy()
                        no_improve = 0
                        # Slightly increase step size if improving? Or keep constant?
                        # Keep constant or slight increase to explore basin
                        step_size = min(step_size * 1.1, 2.0)
                    else:
                        no_improve += 1
                        step_size = max(step_size * 0.8, 1e-4)
                        
                    if no_improve > max_no_improve:
                        # Try a larger jump? Or just break?
                        # If we are stuck, maybe try a random jump from best_x
                        if calls_made + 5 < budget:
                            x_jump = np.clip(best_x + rng.normal(0, 1.0, dim), -5, 5)
                            f_jump = func(x_jump)
                            calls_made += 1
                            if f_jump < best_f:
                                best_f = f_jump
                                best_x = x_jump.copy()
                                x_curr = best_x
                                f_curr = best_f
                                step_size = 0.5
                                no_improve = 0
                            else:
                                break
                        else:
                            break

        except Exception:
            pass

        return best_x, best_f
