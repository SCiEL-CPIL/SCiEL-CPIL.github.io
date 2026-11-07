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
        deadline = start_time + 315
        calls = 0
        max_calls = self.budget
        best_x = None
        best_f = np.inf
        dim = self.dim
        
        # Data storage
        X = []
        Y = []

        def wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= max_calls:
                raise ValueError("Budget Exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time Limit")
            
            calls += 1
            x = np.asarray(x, dtype=float)
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def rbf_kernel(x1, x2, lengthscales, var):
            # x1: (n, d), x2: (n, d)
            # Compute squared distances
            # Using broadcasting: (n, 1, d) - (1, n, d) -> (n, n, d)
            # This can be memory heavy for large n, but n<=500, d=5 is fine.
            # dist_sq = sum((x1[:, None, :] - x2[None, :, :])**2, axis=2)
            # For efficiency, we can compute pairwise distances more carefully if needed, 
            # but for n=500, 500x500x5 array is 1.25M floats ~ 10MB. Acceptable.
            
            # However, standard RBF kernel is exp(-0.5 * sum((x1-x2)^2 / l^2))
            # We can compute this without storing the full (n,n,d) tensor if we are smart,
            # but for simplicity and n=500, let's just do it.
            
            x1_exp = x1[:, np.newaxis, :]
            x2_exp = x2[np.newaxis, :, :]
            diff = x1_exp - x2_exp
            diff_sq = diff ** 2
            # lengthscales shape (d,)
            # We divide each dimension by its lengthscale squared
            ls_sq = lengthscales ** 2
            # diff_sq / ls_sq[None, None, :]
            scaled_dist_sq = np.sum(diff_sq / ls_sq[None, None, :], axis=2)
            K = var * np.exp(-0.5 * scaled_dist_sq)
            return K

        def fit_gp(X, Y):
            n = len(X)
            if n == 1:
                # Simple case
                return np.array([1.0] * dim), 1.0, 0.0

            # Initialize hyperparameters
            # Lengthscales: start at 1.0 for all dims
            # Variance: start at 1.0
            # Noise: start at 1e-6 (assuming noiseless or low noise)
            
            # We will use a few restarts for optimization of hyperparameters
            best_log_lik = -np.inf
            best_ls = np.ones(dim)
            best_var = 1.0
            best_noise = 1e-6
            
            # Heuristic initial values based on data spread
            x_min = np.min(X, axis=0)
            x_max = np.max(X, axis=0)
            ranges = x_max - x_min
            ranges[ranges == 0] = 1.0
            initial_ls = ranges / 2.0
            initial_var = np.var(Y) if n > 1 else 1.0
            if initial_var < 1e-6:
                initial_var = 1.0
            
            # Define objective to maximize (negative for minimization)
            def neg_log_lik(params):
                ls = np.exp(params[:dim]) # Log-scale for positivity
                var = np.exp(params[dim])
                noise = np.exp(params[dim+1])
                
                K = rbf_kernel(X, X, ls, var)
                K += noise * np.eye(n)
                
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    # Add jitter if Cholesky fails
                    K += 1e-6 * np.eye(n)
                    try:
                        L = np.linalg.cholesky(K)
                    except:
                        return 1e6 # Bad log likelihood (large positive for minimization)
                
                # alpha = K^{-1} Y
                # Solve L L^T alpha = Y -> L^T alpha = L^{-1} Y
                v = np.linalg.solve(L, Y)
                alpha = np.linalg.solve(L.T, v)
                
                # Log likelihood: -0.5 * (Y^T K^{-1} Y + log|K| + n log(2pi))
                log_det = 2 * np.sum(np.log(np.diag(L)))
                log_lik = -0.5 * (Y @ alpha + log_det + n * np.log(2 * np.pi))
                
                if not np.isfinite(log_lik):
                    return 1e6
                return -log_lik

            # Optimization using L-BFGS-B
            # Start from heuristic
            x0 = np.concatenate([np.log(initial_ls), [np.log(initial_var)], [np.log(1e-6)]])
            
            # Try a few different starting points
            starts = [
                x0,
                np.concatenate([np.log(np.ones(dim)), [0.0], [-14.0]]), # ls=1, var=1, noise=1e-6
                np.concatenate([np.log(ranges), [np.log(max(1e-2, initial_var))], [-14.0]])
            ]
            
            for x_start in starts:
                try:
                    res = minimize(neg_log_lik, x_start, method='L-BFGS-B', 
                                   bounds=[(0.1, 10.0)]*dim + [( -5, 5), (-20, 0)])
                    if res.fun < best_log_lik:
                        best_log_lik = res.fun
                        best_ls = np.exp(res.x[:dim])
                        best_var = np.exp(res.x[dim])
                        best_noise = np.exp(res.x[dim+1])
                except Exception:
                    pass
            
            # Fallback if all optimizations failed
            if best_log_lik == -np.inf:
                best_ls = initial_ls
                best_var = initial_var
                best_noise = 1e-6

            return best_ls, best_var, best_noise

        def predict_gp(x_new, X, Y, ls, var, noise):
            # x_new: (d,)
            # X: (n, d)
            # Y: (n,)
            
            n = len(X)
            K = rbf_kernel(x_new[None, :], X, ls, var) # (1, n)
            K = K.flatten()
            
            K_ss = var # K(x_new, x_new) is just var (diagonal of RBF is 1*var)
            
            # Compute mean and variance
            # Need K^{-1} Y
            K_full = rbf_kernel(X, X, ls, var) + noise * np.eye(n)
            
            try:
                L = np.linalg.cholesky(K_full)
            except np.linalg.LinAlgError:
                K_full += 1e-6 * np.eye(n)
                L = np.linalg.cholesky(K_full)
            
            v = np.linalg.solve(L, Y)
            alpha = np.linalg.solve(L.T, v)
            
            mean = K @ alpha
            # Var = K_ss - K K_full^{-1} K^T
            # K K_full^{-1} K^T = K (L^{-T} L^{-1}) K^T
            # Let w = L^{-1} K^T
            # Then term = w^T w
            w = np.linalg.solve(L, K)
            var_pred = K_ss - np.dot(w, w)
            if var_pred < 0:
                var_pred = 1e-12
            
            return mean, var_pred

        def expected_improvement(x_new, best_f, mean, var):
            std = np.sqrt(var)
            if std < 1e-10:
                return 0.0
            z = (best_f - mean) / std
            ei = (best_f - mean) * norm.cdf(z) + std * norm.pdf(z)
            return ei

        # 1. Initial Sampling
        # Use Sobol or random. Random is fine for 5D.
        # Let's use a simple random start with a few points to get a baseline.
        n_init = 10
        rng = np.random.default_rng(42)
        X_init = rng.uniform(-5, 5, size=(n_init, dim))
        
        # Evaluate initial points
        for i in range(n_init):
            try:
                f_val = wrapper(X_init[i])
                X.append(X_init[i])
                Y.append(f_val)
            except (ValueError, TimeoutError):
                break

        if len(X) == 0:
            return best_x, best_f

        # 2. Main BO Loop
        while calls < max_calls and time.time() < deadline:
            # Fit GP
            try:
                ls, var, noise = fit_gp(np.array(X), np.array(Y))
            except Exception:
                # If GP fitting fails, fall back to random search
                x_new = rng.uniform(-5, 5, size=dim)
                try:
                    f_val = wrapper(x_new)
                    X.append(x_new)
                    Y.append(f_val)
                except (ValueError, TimeoutError):
                    break
                continue

            # Define acquisition function to maximize
            # EI(x) = (best_f - mu(x)) * Phi(z) + sigma(x) * phi(z)
            
            def neg_ei(x):
                # x is 1D array of size dim
                try:
                    mean, var = predict_gp(x, np.array(X), np.array(Y), ls, var, noise)
                    ei = expected_improvement(x, best_f, mean, var)
                    return -ei
                except Exception:
                    return 1e6

            # Optimize acquisition function
            # Use multiple restarts to avoid local minima in acquisition
            best_x_acq = None
            best_neg_ei = np.inf
            n_restarts = 5
            
            # Start points: best point, random points
            start_points = [np.array(best_x)]
            start_points.extend(rng.uniform(-5, 5, size=(n_restarts-1, dim)))
            
            for start in start_points:
                try:
                    res = minimize(neg_ei, start, method='L-BFGS-B', bounds=[(-5, 5)]*dim)
                    if res.fun < best_neg_ei:
                        best_neg_ei = res.fun
                        best_x_acq = res.x
                except Exception:
                    pass

            if best_x_acq is None:
                # Fallback to random
                best_x_acq = rng.uniform(-5, 5, size=dim)

            # Evaluate new point
            try:
                f_val = wrapper(best_x_acq)
                X.append(best_x_acq)
                Y.append(f_val)
            except (ValueError, TimeoutError):
                break

        if best_x is None:
            best_x = np.zeros(self.dim)
            try:
                best_f = func(best_x)
            except:
                best_f = np.inf

        return best_x, best_f
