"""Rewritten program: uses Gaussian Process regression with L-BFGS-B hyperparameter optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, cholesky

warnings.filterwarnings('ignore')

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 285  # Safety margin for 330s limit
    
    X = np.asarray(X_train, dtype=float).reshape(-1, 1)
    y = np.asarray(y_train, dtype=float).flatten()
    n = len(y)
    
    if n < 2:
        mean_y = np.mean(y) if n > 0 else 0.0
        return lambda X: np.full(len(np.asarray(X).reshape(-1)), mean_y)
    
    x_vals = X[:, 0].copy()
    y_vals = y.copy()
    
    # Check if y is constant
    if np.std(y_vals) < 1e-12:
        mean_y = np.mean(y_vals)
        return lambda X: np.full(len(np.asarray(X).reshape(-1)), mean_y)
    
    y_var = np.var(y_vals)
    if y_var < 1e-15:
        mean_y = np.mean(y_vals)
        return lambda X: np.full(len(np.asarray(X).reshape(-1)), mean_y)
    
    # Center and scale x for better numerical conditioning
    x_mean = np.mean(x_vals)
    x_std = np.std(x_vals)
    if x_std < 1e-10:
        x_std = 1.0
    x_scaled = (x_vals - x_mean) / x_std
    
    # Normalize y for better conditioning
    y_mean = np.mean(y_vals)
    y_std = np.std(y_vals)
    if y_std < 1e-10:
        y_std = 1.0
    y_scaled = (y_vals - y_mean) / y_std
    
    # GP kernel: squared exponential (RBF)
    # K(x, x') = sigma_f^2 * exp(-0.5 * (x - x')^2 / l^2)
    # For noiseless data, we add tiny noise for numerical stability
    
    def rbf_kernel(X1, X2, length_scale, signal_var):
        """Compute RBF kernel matrix between X1 and X2."""
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        # Compute squared distances
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
        sq_dists = np.maximum(sq_dists, 0)
        return signal_var * np.exp(-0.5 * sq_dists / (length_scale ** 2))
    
    def log_marginal_likelihood(params):
        """Negative log marginal likelihood for GP with RBF kernel."""
        if time.time() > deadline - 3:
            return 1e10  # Penalize if we're out of time
        
        length_scale = np.exp(params[0])  # Log-transform for positivity
        signal_var = np.exp(params[1])
        noise_var = np.exp(params[2])
        
        # Clamp to reasonable ranges
        if length_scale < 1e-3 or length_scale > 1e3:
            return 1e10
        if signal_var < 1e-6 or signal_var > 1e6:
            return 1e10
        if noise_var < 1e-10 or noise_var > 1e4:
            return 1e10
        
        try:
            K = rbf_kernel(x_scaled, x_scaled, length_scale, signal_var)
            K[np.diag_indices_from(K)] += noise_var
            
            # Cholesky decomposition
            try:
                L = cholesky(K, lower=True)
            except Exception:
                # Add more noise if Cholesky fails
                K[np.diag_indices_from(K)] += 1e-6
                try:
                    L = cholesky(K, lower=True)
                except Exception:
                    return 1e10
            
            # Compute alpha = K^{-1} y
            alpha = cho_solve((L, True), y_scaled)
            
            # Negative log marginal likelihood
            nll = 0.5 * np.dot(y_scaled, alpha) + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
            
            if not np.isfinite(nll):
                return 1e10
            
            return nll
        
        except Exception:
            return 1e10
    
    # Optimize hyperparameters using L-BFGS-B
    # Initial guesses: length_scale ~ 1.0, signal_var ~ 1.0, noise_var ~ 1e-8
    # Work in log-space for optimization
    x0 = np.array([0.0, 0.0, -18.0])  # log(length_scale), log(signal_var), log(noise_var)
    
    best_nll = np.inf
    best_params = x0.copy()
    
    # Try multiple random starts
    np.random.seed(42)
    n_starts = 5
    for s in range(n_starts):
        if time.time() > deadline - 10:
            break
        
        if s == 0:
            start = x0.copy()
        else:
            start = x0 + np.random.randn(3) * 1.5
            start[2] = -18.0 + np.random.randn() * 2  # Keep noise small
        
        try:
            result = minimize(
                log_marginal_likelihood,
                start,
                method='L-BFGS-B',
                bounds=[(-3, 3), (-3, 3), (-25, -5)],
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
        except Exception:
            continue
    
    # Extract optimal hyperparameters
    opt_length_scale = np.exp(best_params[0])
    opt_signal_var = np.exp(best_params[1])
    opt_noise_var = np.exp(best_params[2])
    
    # Clamp to reasonable values
    opt_length_scale = np.clip(opt_length_scale, 1e-3, 1e3)
    opt_signal_var = np.clip(opt_signal_var, 1e-6, 1e6)
    opt_noise_var = np.clip(opt_noise_var, 1e-10, 1e4)
    
    # Compute final kernel matrix and Cholesky
    K = rbf_kernel(x_scaled, x_scaled, opt_length_scale, opt_signal_var)
    K[np.diag_indices_from(K)] += opt_noise_var
    
    try:
        L = cholesky(K, lower=True)
        L_factor = (L, True)
    except Exception:
        K[np.diag_indices_from(K)] += 1e-6
        try:
            L = cholesky(K, lower=True)
            L_factor = (L, True)
        except Exception:
            # Fallback: use mean prediction
            mean_y = np.mean(y_vals)
            return lambda X: np.full(len(np.asarray(X).reshape(-1)), mean_y)
    
    # Compute alpha = K^{-1} y
    alpha = cho_solve(L_factor, y_scaled)
    
    # Create prediction function
    def predict(X_pred):
        X_pred = np.asarray(X_pred, dtype=float).reshape(-1, 1)[:, 0]
        n_pred = len(X_pred)
        
        # Scale input
        x_pred_scaled = (X_pred - x_mean) / x_std
        
        # Compute kernel between test and train
        K_s = rbf_kernel(x_pred_scaled, x_scaled, opt_length_scale, opt_signal_var)
        
        # Prediction: y_pred = K_s @ alpha
        y_pred_scaled = K_s @ alpha
        
        # Unscale
        y_pred = y_pred_scaled * y_std + y_mean
        
        # Ensure finite values
        y_pred = np.where(np.isfinite(y_pred), y_pred, y_mean)
        
        return y_pred
    
    return predict
# EVOLVE-BLOCK-END
