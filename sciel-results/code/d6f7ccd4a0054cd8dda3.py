"""
Strategy: 
1. Use Gaussian Process Regression (GPR) to denoise the training data and capture the underlying smooth structure of the subgrid tendency.
2. Project the GPR predictions onto a degree-5 polynomial basis using Ridge Regression (Tikhonov regularization) to ensure stability and generalization.
3. This avoids the ill-conditioning of direct high-degree polynomial fits and the overfitting of unregularized GPR, while respecting the mandatory polynomial output format.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s limit
    
    x = np.asarray(training_x, dtype=float)
    y = np.asarray(training_subgrid_tendency, dtype=float)
    
    # Ensure 1D
    x = x.ravel()
    y = y.ravel()
    
    n_samples = len(x)
    
    # Basic sanity checks
    if n_samples < 10:
        # Fallback to linear fit if very few samples
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            return np.pad(coeff, (0, 6 - len(coeff)))
        except:
            return np.zeros(6)
    
    # 1. Gaussian Process Regression to get smooth predictions
    # We use a Matern kernel for smoothness. 
    # Length scale and signal variance are set heuristically to be robust.
    try:
        from scipy.stats import norm
        from scipy.optimize import minimize
        from scipy.linalg import cho_factor, cho_solve
    except ImportError:
        # Fallback if scipy is not available (unlikely per contract)
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 3)
            return np.pad(coeff, (0, 6 - len(coeff)))
        except:
            return np.zeros(6)

    # Heuristic length scale: based on data spread
    x_range = np.max(x) - np.min(x)
    if x_range == 0:
        x_range = 1.0
    l0 = x_range / 5.0
    
    # Signal variance: variance of y
    y_var = np.var(y)
    if y_var == 0:
        y_var = 1.0
    sig_var = y_var
    
    # Noise variance: small fraction of signal to allow some flexibility
    # We estimate noise as a small fraction to prevent overfitting
    noise_var = 0.01 * y_var
    
    # Matern kernel (nu=2.5 for smoothness)
    def matern_kernel(X1, X2, l, sig):
        X1 = X1[:, np.newaxis]
        X2 = X2[np.newaxis, :]
        dist = np.sqrt((X1 - X2)**2)
        # Avoid division by zero
        dist = np.maximum(dist, 1e-10)
        k = (sig**2) * (1 + np.sqrt(5)*dist/l) * np.exp(-np.sqrt(5)*dist/l)
        return k

    # We will fit the GPR parameters (l, sig_var, noise_var) using a quick optimization
    # to maximize log marginal likelihood. 
    # To save time, we might just use the heuristics if time is tight.
    
    # Let's try a quick optimization if time permits, otherwise use heuristics.
    # Optimization can be expensive, so we limit it.
    
    def log_marginal_likelihood(neg_log_params):
        if time.time() > deadline - 10:
            return 0 # Stop optimizing
            
        log_l, log_sig, log_noise = neg_log_params
        l = np.exp(log_l)
        sig = np.exp(log_sig)
        noise = np.exp(log_noise)
        
        K = matern_kernel(x, x, l, sig) + noise * np.eye(n_samples)
        
        try:
            L, info = cho_factor(K)
            if info < 0:
                return -1e10
            alpha = cho_solve((L, info), y)
            log_det = 2 * np.sum(np.log(np.diagonal(L)))
            ll = -0.5 * y @ alpha - 0.5 * log_det - 0.5 * n_samples * np.log(2 * np.pi)
            return -ll # Minimize negative log likelihood
        except:
            return 1e10

    # Initial guess in log space
    x0 = [np.log(l0), np.log(sig_var), np.log(noise_var)]
    
    # Try to optimize for a short time
    best_params = x0
    best_ll = 1e10
    
    try:
        # Use Nelder-Mead for robustness
        res = minimize(log_marginal_likelihood, x0, method='Nelder-Mead', 
                       options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-3})
        
        if res.fun < best_ll and time.time() < deadline - 5:
            best_params = res.x
    except:
        pass
    
    l_opt = np.exp(best_params[0])
    sig_opt = np.exp(best_params[1])
    noise_opt = np.exp(best_params[2])
    
    # 2. Project the GPR mean function onto the polynomial basis
    # Instead of using raw y (noisy), we use the GPR mean predictions at the training points
    # as the target for the polynomial fit. This acts as a strong denoiser.
    
    K = matern_kernel(x, x, l_opt, sig_opt) + noise_opt * np.eye(n_samples)
    
    try:
        L, info = cho_factor(K)
        if info < 0:
            raise ValueError("Cholesky failed")
        alpha = cho_solve((L, info), y)
        y_smooth = K @ alpha
    except:
        # Fallback to raw y if GPR fails
        y_smooth = y
    
    # Now fit degree-5 polynomial to (x, y_smooth)
    # Use Ridge Regression to handle ill-conditioning of the Vandermonde matrix
    # Design matrix V
    V = np.vander(x, N=6, increasing=True)
    
    # Tikhonov regularization
    # Choose lambda based on matrix condition number or heuristically
    # A small lambda is sufficient because y_smooth is already denoised
    lambda_reg = 1e-4 * np.trace(V.T @ V) / n_samples
    
    A = V.T @ V + lambda_reg * np.eye(6)
    b = V.T @ y_smooth
    
    try:
        coeff = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular
        try:
            coeff = np.linalg.lstsq(V, y_smooth, rcond=None)[0]
        except:
            coeff = np.zeros(6)
            
    # Ensure coefficients are finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)
        
    return coeff
# EVOLVE-BLOCK-END
