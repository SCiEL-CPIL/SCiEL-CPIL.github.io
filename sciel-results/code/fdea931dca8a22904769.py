"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """
    Bayesian Optimization with Gaussian Process surrogate.
    1. Initial exploration using a quasi-random Halton sequence.
    2. Iterative refinement using a GP model and Expected Improvement acquisition.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s kill
        
        dim = self.dim
        budget = self.budget
        
        # State
        X = []  # List of evaluated points
        Y = []  # List of evaluated values
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def eval_point(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            X.append(x)
            Y.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Halton sequence for initial diverse sampling
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # Phase 1: Initial Exploration
        # For dim=5, a small number of well-spread points helps establish the landscape.
        # 20-30 points is usually enough to find at least one good basin among 101 peaks.
        n_init = 25
        for i in range(1, n_init + 1):
            point = np.array([halton(i, base) for base in range(2, dim + 2)])
            point = point * 10.0 - 5.0  # Map to [-5, 5]
            eval_point(point)
            
            if calls_made >= budget or time.time() > deadline:
                break

        if calls_made >= budget:
            return best_x, best_f

        # Phase 2: Bayesian Optimization Loop
        # We will fit a GP to the data and use EI to select the next point.
        
        def fit_gp(X_train, Y_train):
            """
            Fit a simple Gaussian Process with RBF kernel.
            Returns a function that predicts mean and variance.
            """
            X_train = np.array(X_train)
            Y_train = np.array(Y_train)
            
            # Standardize Y for numerical stability in GP
            y_mean = np.mean(Y_train)
            y_std = np.std(Y_train)
            if y_std < 1e-10:
                y_std = 1.0
            Y_std = (Y_train - y_mean) / y_std
            
            n = len(X_train)
            d = X_train.shape[1]
            
            # Kernel parameters: length_scale, signal_variance, noise_variance
            # We use fixed reasonable parameters for speed, or simple optimization if time permits.
            # Given the tight budget and potential for many GP fits, fixed params or simple heuristic is safer.
            # Let's try to optimize length_scale quickly if n is small, otherwise use heuristic.
            
            # Heuristic length scale based on best distance
            # Or just fix it. For dim=5, a length scale of 1.0 to 2.0 is often reasonable.
            # Let's use a simple heuristic: length_scale = 1.0
            # signal_variance = 1.0 (since Y is standardized)
            # noise_variance = 1e-6
            
            l = 1.0
            sigma_f_sq = 1.0
            sigma_n_sq = 1e-6
            
            # Compute kernel matrix K
            # K_ij = sigma_f_sq * exp(-0.5 * ||x_i - x_j||^2 / l^2)
            # Efficient computation for small n
            diff = X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2)
            K = sigma_f_sq * np.exp(-0.5 * sq_dist / (l**2))
            K += sigma_n_sq * np.eye(n)
            
            # Solve for alpha: K alpha = Y_std
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_std))
            except np.linalg.LinAlgError:
                # Add jitter if Cholesky fails
                K += 1e-6 * np.eye(n)
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_std))
                except:
                    # Fallback: return current best
                    return None

            def predict(X_new):
                X_new = np.array(X_new)
                if X_new.ndim == 1:
                    X_new = X_new[np.newaxis, :]
                # k(X_new, X_train)
                diff = X_new[:, np.newaxis, :] - X_train[np.newaxis, :, :]
                sq_dist = np.sum(diff**2, axis=2)
                k = sigma_f_sq * np.exp(-0.5 * sq_dist / (l**2))
                
                mean = k @ alpha * y_std + y_mean
                # Variance calculation (simplified, ignoring prior variance term for ranking if n is large enough)
                # var(X_new) = k(X_new, X_new) - k(X_new, X_train) @ K^-1 @ k(X_train, X_new)
                # For EI, we need std.
                v = np.linalg.solve(L, k.T)
                var = sigma_f_sq - np.sum(v * k, axis=1)
                # Clip negative variance due to numerical errors
                var = np.clip(var, 1e-10, None)
                std = np.sqrt(var)
                
                return mean, std

            return predict

        # Acquisition Function: Expected Improvement
        def expected_improvement(X_cand, current_best_f, predict_func):
            """
            X_cand: (n, dim) array of candidate points
            current_best_f: current best objective value
            predict_func: function returning mean and std for X_cand
            """
            if predict_func is None:
                return np.zeros(len(X_cand))
            
            mean, std = predict_func(X_cand)
            # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
            # where Z = (f_best - mu) / sigma
            # Note: We are minimizing.
            # If mu < f_best, Z > 0, EI is positive.
            
            with np.errstate(divide='ignore', invalid='ignore'):
                Z = (current_best_f - mean) / std
                # Handle std=0
                Z = np.where(std < 1e-10, 0, Z)
                
                EI = (current_best_f - mean) * norm.cdf(Z) + std * norm.pdf(Z)
            
            return EI

        # Local optimization of the acquisition function
        def optimize_acquisition(predict_func, current_best_f):
            # Use multiple random starts for the local minimization of -EI
            rng = np.random.default_rng(42)
            
            # If we have very few points, random search might be better, 
            # but let's try to optimize EI.
            
            n_starts = 10
            best_x_acq = None
            best_ei = -np.inf
            
            # Initial guesses: current best point, and random points
            start_points = [best_x.copy()]
            for _ in range(n_starts - 1):
                start_points.append(rng.uniform(-5, 5, dim))
            
            for x0 in start_points:
                # Objective: minimize -EI
                def neg_ei(x):
                    x = np.clip(x, -5, 5)
                    ei = expected_improvement(x.reshape(1, -1), current_best_f, predict_func)[0]
                    return -ei
                
                try:
                    res = minimize(neg_ei, x0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6})
                    if res.success or res.fun < best_ei:
                        # Check if it's a valid improvement
                        x_opt = np.clip(res.x, -5, 5)
                        ei_val = -res.fun
                        if ei_val > best_ei:
                            best_ei = ei_val
                            best_x_acq = x_opt
                except Exception:
                    continue
            
            if best_x_acq is None:
                # Fallback to random
                best_x_acq = rng.uniform(-5, 5, dim)
                
            return best_x_acq

        # Main BO Loop
        while calls_made < budget:
            if time.time() > deadline - 5.0:  # Leave 5s for final checks
                break
            
            # Fit GP
            predict_func = fit_gp(X, Y)
            
            # If GP fitting failed, fall back to random search
            if predict_func is None:
                x_new = np.random.uniform(-5, 5, dim)
            else:
                # Optimize acquisition function
                x_new = optimize_acquisition(predict_func, best_f)
            
            # Evaluate new point
            eval_point(x_new)
            
            # Simple early stopping if we seem to have converged
            # (Optional, but helps save time if stuck)
            # For now, rely on budget and time.

        return best_x, best_f
# EVOLVE-BLOCK-END
