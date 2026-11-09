import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """
    Surrogate-Assisted Bayesian Optimization with Local Refinement for BBOB f21.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        n = dim

        best_x = None
        best_f = np.inf
        call_count = 0

        # Storage for Bayesian Optimization
        X = []
        Y = []

        def check_time():
            return time.time() > deadline - 5

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # --- Phase 1: Space-Filling Initialization (Halton Sequence) ---
        # Halton sequence provides better low-discrepancy coverage than random
        def halton_sequence(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton(dim, n_points):
            # Generate Halton points in [0, 1]^dim
            points = np.zeros((n_points, dim))
            for i in range(n_points):
                for j in range(dim):
                    # Use prime bases for each dimension
                    base = 2 ** (j + 1)  # 2, 4, 8, 16, 32
                    # Standard Halton uses primes: 2, 3, 5, 7, 11
                    # Let's use actual primes for better properties
                    primes = [2, 3, 5, 7, 11, 13]
                    points[i, j] = halton_sequence(i + 1, primes[j])
            return points

        n_init = 40
        # Generate Halton points in [0, 1]^d and scale to [-5, 5]^d
        halton_pts = halton(n, n_init)
        init_points = halton_pts * 10.0 - 5.0

        for sp in init_points:
            if call_count >= budget or check_time():
                break
            x = np.clip(sp, -5, 5)
            fx = func(x)
            call_count += 1
            X.append(x)
            Y.append(fx)
            update_best(x, fx)

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            call_count += 1
            X.append(best_x)
            Y.append(best_f)

        # --- Phase 2: Bayesian Optimization Loop ---
        # We will run BO until we have used ~60% of the budget or time runs out
        bo_budget_limit = int(budget * 0.6)
        
        # GP Parameters
        # We use a simple GP with Matern 5/2 kernel
        # To keep it fast, we don't fit hyperparameters every step, or we fit them quickly.
        # For n=5 and <300 points, fitting hyperparameters is feasible.
        
        def matern52_kernel(X1, X2, length_scale):
            """Compute Matern 5/2 kernel between X1 and X2."""
            # X1: (n1, d), X2: (n2, d)
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2)) / length_scale
            return (1 + np.sqrt(5) * dist + 5/3 * dist**2) * np.exp(-np.sqrt(5) * dist)

        def gp_predict(X_train, Y_train, X_test, length_scale, sigma_f, sigma_n):
            """
            Gaussian Process Prediction.
            Returns mean and variance.
            """
            K = matern52_kernel(X_train, X_train, length_scale) + sigma_n * np.eye(len(X_train))
            K_star = matern52_kernel(X_train, X_test, length_scale)
            
            # Solve K alpha = y
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            except np.linalg.LinAlgError:
                # Fallback to lstsq if Cholesky fails
                alpha = np.linalg.lstsq(K, Y_train, rcond=None)[0]
                
            mean = K_star.T @ alpha
            v = np.linalg.solve(L, K_star.T) if 'L' in locals() else np.linalg.lstsq(K, K_star.T, rcond=None)[0]
            var = sigma_f**2 - np.sum(v**2, axis=0)
            var = np.maximum(var, 0)
            
            return mean, var

        def expected_improvement(X, Y, best_f, length_scale, sigma_f, sigma_n):
            """
            Compute Expected Improvement for a set of candidate points.
            """
            mean, var = gp_predict(np.array(X), np.array(Y), X, length_scale, sigma_f, sigma_n)
            std = np.sqrt(var)
            z = (best_f - mean) / (std + 1e-10)
            ei = (best_f - mean) * norm.cdf(z) + std * norm.pdf(z)
            return ei

        def optimize_acquisition(X, Y, best_f, length_scale, sigma_f, sigma_n):
            """
            Find the point that maximizes EI.
            Uses multiple random starts + Nelder-Mead.
            """
            def neg_ei(x):
                x = np.clip(x, -5, 5)
                ei = expected_improvement(X, Y, best_f, length_scale, sigma_f, sigma_n, np.array([x]))
                return -ei[0]

            # Try a few random starts for the optimizer
            best_x_opt = None
            best_val = np.inf
            
            n_starts = 10
            for i in range(n_starts):
                if i == 0:
                    x0 = best_x.copy()
                else:
                    x0 = rng.uniform(-5, 5, dim)
                
                res = minimize(neg_ei, x0, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6})
                if res.fun < best_val:
                    best_val = res.fun
                    best_x_opt = res.x
            
            return best_x_opt

        # Adaptive Hyperparameters for GP
        # Initial guess: length_scale ~ 2.0, sigma_f ~ std(Y), sigma_n ~ 1e-4
        current_length_scale = 2.0
        current_sigma_f = max(np.std(Y) if len(Y) > 1 else 1.0, 0.1)
        current_sigma_n = 1e-4

        # BO Loop
        while call_count < bo_budget_limit and not check_time():
            # 1. Fit GP Hyperparameters (simplified: just adapt length_scale based on spread)
            # For speed, we might skip full MLE and use a heuristic or fixed params.
            # Given n=5 and small N, let's do a quick grid search on length_scale if we have time.
            # Actually, to be safe on time, let's use a fixed reasonable length scale 
            # or update it slowly. F21 has peaks separated by ~1-2 units.
            
            # Heuristic update: if we are stuck, increase exploration (larger length scale? No, smaller LS = more local, larger LS = smoother).
            # Let's stick to a robust fixed LS for now, or simple adaptation.
            # A common trick: LS = 1.0 / (mean distance between points)
            if len(X) > 1:
                X_arr = np.array(X)
                # Sample a few pairs to estimate typical distance
                idx = rng.choice(len(X_arr), size=min(100, len(X_arr)), replace=False)
                d = np.linalg.norm(X_arr[idx][:, np.newaxis, :] - X_arr[idx][np.newaxis, :, :], axis=2)
                mean_dist = np.mean(d[d > 1e-6])
                # Map mean distance to length scale
                current_length_scale = max(0.5, min(5.0, mean_dist * 0.8))

            # 2. Optimize Acquisition Function
            x_next = optimize_acquisition(X, Y, best_f, current_length_scale, current_sigma_f, current_sigma_n)
            
            if x_next is None:
                # Fallback to random
                x_next = rng.uniform(-5, 5, dim)
            
            x_next = np.clip(x_next, -5, 5)
            
            # 3. Evaluate
            fx = func(x_next)
            call_count += 1
            X.append(x_next)
            Y.append(fx)
            update_best(x_next, fx)
            
            # 4. Check for stagnation in BO
            # If we haven't improved best_f in last 15 iterations, 
            # it might be good to switch to local search earlier.
            # We'll track this simply by checking if we are close to budget limit.
            
            # Optional: Update sigma_f to match current scale
            if len(Y) > 10:
                current_sigma_f = np.std(Y[-10:])
                current_sigma_f = max(current_sigma_f, 0.01)

        # --- Phase 3: Local Refinement ---
        # Use remaining budget for a fast local search from the best point found by BO.
        # (1+1)-ES is very fast and effective for final convergence.
        
        if call_count < budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5 # Start smaller than global search
            stagnation = 0
            max_stagnation = 30
            
            # Adaptive (1+1)-ES
            while call_count < budget and not check_time():
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = func(y)
                call_count += 1
                
                if fy < fx - 1e-10:
                    x = y
                    fx = fy
                    sigma *= 1.2
                    stagnation = 0
                    update_best(x, fx)
                else:
                    sigma *= 0.85
                    stagnation += 1
                    if stagnation >= max_stagnation:
                        # Restart local search with a larger step or random jump?
                        # If we are stuck, maybe we are in a flat area or at the boundary.
                        # Let's just keep going, sigma will shrink.
                        # If sigma becomes too small, we can break.
                        if sigma < 1e-4:
                            break
                        stagnation = 0
                        # Try a larger jump to escape potential local minima if not global
                        # But since BO should have found the basin, we assume we are close to the peak's minimum.
                        
                sigma = max(sigma, 1e-5)
                sigma = min(sigma, 5.0)

        return best_x, best_f
