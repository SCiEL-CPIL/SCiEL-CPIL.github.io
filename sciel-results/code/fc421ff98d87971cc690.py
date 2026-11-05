"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing.
    
    Strategy:
    1. Use Gaussian Process Regression (GPR) to model the relationship between x and y.
       GPR provides a flexible non-parametric fit that can capture local structures
       better than a global polynomial, while the kernel choice (RBF) ensures smoothness.
    2. Since the output must be a degree-5 polynomial, we project the GP predictions
       onto the monomial basis [1, x, x^2, x^3, x^4, x^5] using least squares.
    3. To improve the projection, we evaluate the GP at a large number of points
       spanning the range of x, creating a dense set of (x, y_gp) pairs.
    4. Fit the degree-5 polynomial to these dense GP predictions.
    5. This approach leverages the GP's ability to adapt to the data's local variability
       while constraining the final output to the required polynomial form, which helps
       with generalization to unseen regimes by avoiding overfitting to specific training points
       that a direct polynomial fit might latch onto.
    """
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Remove any potential NaNs/Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) == 0:
        return np.zeros(6)

    # 1. Prepare data for GP
    # Normalize x for better GP performance
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    
    if x_range < 1e-12:
        return np.array([np.mean(y), 0, 0, 0, 0, 0])
    
    # Standardize x to [0, 1] for kernel scaling
    x_norm = (x - x_min) / x_range
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    # 2. Define GP Kernel and Model
    # We use a simple RBF kernel with a linear trend.
    # K(x, x') = sigma_f^2 * exp(-0.5 * ||x - x'||^2 / l^2) + sigma_n^2 * I + linear_trend
    
    # To avoid heavy dependencies or complex optimization, we implement a simplified GP
    # or use scipy's minimize to find optimal length scale l and variance sigma_f.
    
    N = len(x_norm)
    # Design matrix for linear trend (constant and linear term)
    X_design = np.column_stack([np.ones(N), x_norm])
    
    def log_marginal_likelihood(params):
        log_l, log_sigma_f, log_sigma_n = params
        l = np.exp(log_l)
        sigma_f = np.exp(log_sigma_f)
        sigma_n = np.exp(log_sigma_n)
        
        # Compute kernel matrix
        # For large N, this can be expensive. If N is very large, subsample.
        # Given the task, N is likely manageable (flattened trajectories).
        # Let's assume N < 10000 for direct computation. If larger, we should subsample.
        if N > 5000:
            idx = np.random.choice(N, 5000, replace=False)
            x_sub = x_norm[idx]
            y_sub = y_norm[idx]
            X_sub = X_design[idx]
        else:
            x_sub = x_norm
            y_sub = y_norm
            X_sub = X_design
            
        # RBF Kernel
        diff = x_sub[:, None] - x_sub[None, :]
        K = sigma_f**2 * np.exp(-0.5 * diff**2 / l**2)
        
        # Add noise
        K += sigma_n**2 * np.eye(len(x_sub))
        
        # Add linear trend covariance? 
        # Standard GP with mean function. We can subtract the linear trend first.
        # Let's simplify: Assume zero mean, but the linear trend is captured by the kernel if l is large.
        # Or, we can center y and subtract the linear fit.
        # Let's do a simpler approach: Fit a linear trend first, then GP on residuals.
        
        # Linear fit on normalized data
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(X_sub, y_sub, rcond=None)
            y_pred_linear = X_sub @ coeffs
            y_resid = y_sub - y_pred_linear
        except:
            return 1e10 # Bad params
        
        # GP on residuals
        K_resid = K # Kernel doesn't change
        # Solve K @ alpha = y_resid
        try:
            alpha = np.linalg.solve(K_resid, y_resid)
            L = np.linalg.cholesky(K_resid)
            v = np.linalg.solve(L, y_resid)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            mll = -0.5 * (y_resid @ alpha + log_det + len(y_resid) * np.log(2 * np.pi))
            return -mll
        except:
            return 1e10

    # Optimize hyperparameters
    # Initial guesses
    log_l = np.log(0.5)
    log_sigma_f = np.log(1.0)
    log_sigma_n = np.log(0.1)
    
    # Use Nelder-Mead for robustness without gradient
    try:
        res = minimize(log_marginal_likelihood, [log_l, log_sigma_f, log_sigma_n], 
                       method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 100})
        best_params = res.x
        log_l, log_sigma_f, log_sigma_n = best_params
    except:
        log_l, log_sigma_f, log_sigma_n = [np.log(0.5), np.log(1.0), np.log(0.1)]

    l = np.exp(log_l)
    sigma_f = np.exp(log_sigma_f)
    sigma_n = np.exp(log_sigma_n)

    # 3. Generate dense predictions
    # Create a grid of x points to predict on
    num_points = 2000
    x_grid = np.linspace(x_min, x_max, num_points)
    x_grid_norm = (x_grid - x_min) / x_range
    
    # Predict GP values on grid
    # We need the mean function (linear trend) + GP residual
    X_grid_design = np.column_stack([np.ones(num_points), x_grid_norm])
    
    # Recalculate linear coeffs on full data (or subsampled if N was large)
    if N > 5000:
        # Use the same subsample for consistency? No, better to use full data for linear fit if possible
        # But if N is huge, lstsq is slow. Let's stick to the subsample used in optimization for consistency
        # or just re-fit on full data if time permits.
        # For now, let's re-fit on full data for the linear part to be accurate.
        try:
            coeffs_full, _, _, _ = np.linalg.lstsq(X_design, y_norm, rcond=None)
            y_mean_grid = X_grid_design @ coeffs_full
        except:
            y_mean_grid = np.zeros(num_points)
    else:
        try:
            coeffs_full, _, _, _ = np.linalg.lstsq(X_design, y_norm, rcond=None)
            y_mean_grid = X_grid_design @ coeffs_full
        except:
            y_mean_grid = np.zeros(num_points)

    # GP residual prediction
    # K(x_grid, x_train) @ alpha
    # alpha was computed on the subsampled data if N > 5000.
    # This is a mismatch. To be safe and simple, let's re-compute alpha on the full data 
    # if N is not too large, or just use the linear fit if GP is too complex.
    
    # Given the complexity and time limits, a simpler approach might be better:
    # Fit a polynomial directly but with regularization?
    # The prompt suggests "piecewise cubic spline" or "GPR".
    # Let's try a simpler GPR implementation that doesn't subsample, but handles N efficiently.
    
    # If N is large, GPR is O(N^3). 120s limit.
    # Let's check N.
    if N > 2000:
        # Subsample for GP to keep it fast
        idx = np.random.choice(N, 2000, replace=False)
        x_train_gp = x_norm[idx]
        y_train_gp = y_norm[idx]
        X_train_gp = X_design[idx]
    else:
        x_train_gp = x_norm
        y_train_gp = y_norm
        X_train_gp = X_design

    # Fit linear trend on training set
    coeffs_train, _, _, _ = np.linalg.lstsq(X_train_gp, y_train_gp, rcond=None)
    y_resid_train = y_train_gp - X_train_gp @ coeffs_train
    
    # Compute Kernel on training set
    diff_train = x_train_gp[:, None] - x_train_gp[None, :]
    K_train = sigma_f**2 * np.exp(-0.5 * diff_train**2 / l**2)
    K_train += sigma_n**2 * np.eye(len(x_train_gp))
    
    try:
        alpha = np.linalg.solve(K_train, y_resid_train)
    except:
        alpha = np.zeros(len(x_train_gp))

    # Predict on grid
    diff_grid = x_grid_norm[:, None] - x_train_gp[None, :]
    K_grid_train = sigma_f**2 * np.exp(-0.5 * diff_grid**2 / l**2)
    
    y_gp_residual_grid = K_grid_train @ alpha
    y_gp_grid = y_mean_grid + y_gp_residual_grid
    
    # Denormalize y
    y_gp_grid = y_gp_grid * y_std + y_mean

    # 4. Fit degree-5 polynomial to GP predictions
    # We want to find coeffs c such that sum(c_i * x^i) approximates y_gp_grid
    # Use np.polyfit? It returns descending powers.
    # Better to use np.linalg.lstsq with Vandermonde matrix for ascending powers.
    
    # Construct Vandermonde matrix for x_grid
    V = np.vander(x_grid, 6, increasing=True)
    
    try:
        coeffs_poly, residuals, rank, sv = np.linalg.lstsq(V, y_gp_grid, rcond=None)
    except:
        # Fallback to simple polyfit on original data
        try:
            coeffs_desc = np.polyfit(x, y, 5)
            coeffs_poly = coeffs_desc[::-1]
        except:
            coeffs_poly = np.zeros(6)
            coeffs_poly[0] = np.mean(y)

    # Check time
    if time.time() - start_time > DEADLINE:
        pass

    # Final sanity check
    if not np.all(np.isfinite(coeffs_poly)):
        coeffs_poly = np.zeros(6)
        coeffs_poly[0] = np.mean(y)

    return coeffs_poly
# EVOLVE-BLOCK-END
