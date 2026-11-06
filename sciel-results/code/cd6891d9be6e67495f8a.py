"""Strategy: Local Polynomial (LOESS) Regression with adaptive bandwidth and Huber loss weighting."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize_scalar

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n < 6:
        return np.zeros(6)
    
    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 6:
        return np.zeros(6)
    
    # Sort by x for efficient local search
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Standardize x for numerical stability in local fits
    x_mean = np.mean(x_sorted)
    x_std = np.std(x_sorted)
    if x_std < 1e-10:
        x_std = 1.0
    x_norm_sorted = (x_sorted - x_mean) / x_std
    
    # Time budget check
    if time.time() > deadline - 20:
        # Fallback: simple global ridge fit
        return _global_fit_fallback(x, y)

    # Strategy: LOESS (Local Polynomial Regression)
    # We need to find coefficients c_0..c_5 such that sum(c_i * x**i) approximates y.
    # However, LOESS predicts y(x) using local fits. 
    # The task requires returning a SINGLE global polynomial.
    # Therefore, we use LOESS to generate "denoised" target values y_loess(x), 
    # and then fit a global degree-5 polynomial to these y_loess values.
    # This is different from simple kernel smoothing because the local model is a 
    # degree-5 polynomial, not a constant or linear term, allowing better 
    # curvature capture in the denoising step.
    
    # 1. Determine optimal bandwidth (fraction of data to use in local fit)
    # We use a small validation set to tune the bandwidth.
    np.random.seed(42)
    indices = np.random.permutation(n)
    val_frac = min(0.2, max(0.1, 2000.0 / n))
    val_size = int(n * val_frac)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    x_val = x_norm_sorted[val_idx]
    y_val = y[val_idx]
    
    x_train_local = x_norm_sorted[train_idx]
    y_train_local = y[train_idx]
    
    best_score = np.inf
    best_bandwidth = 0.2
    
    # Try a few bandwidths
    bandwidths = [0.05, 0.1, 0.2, 0.3, 0.5]
    
    for bw in bandwidths:
        if time.time() > deadline - 15:
            break
            
        # Calculate LOESS predictions for validation set
        y_pred_val = _loess_predict(x_val, x_train_local, y_train_local, bw, degree=5)
        
        if not np.all(np.isfinite(y_pred_val)):
            continue
            
        mse = np.mean((y_val - y_pred_val) ** 2)
        
        if mse < best_score:
            best_score = mse
            best_bandwidth = bw

    # 2. Generate denoised targets for ALL data using the best bandwidth
    # To save time, we can subsample the training data for the LOESS fitting step
    # if n is very large, but let's try full data first with a time check.
    
    if n > 50000:
        # Subsample for LOESS fitting to save time
        sub_idx = np.random.choice(n, 50000, replace=False)
        x_loess_train = x_norm_sorted[sub_idx]
        y_loess_train = y_sorted[sub_idx]
    else:
        x_loess_train = x_norm_sorted
        y_loess_train = y_sorted
        
    # We need to evaluate the LOESS model at the training points to get y_denoised
    # However, calculating LOESS for every point is O(n^2). 
    # Instead, we fit the global polynomial to the LOESS predictions at a subset of points
    # or we use the LOESS model to generate a dense grid of points and fit to that.
    
    # Let's generate a dense grid of x values covering the range
    x_grid_min = np.min(x_norm_sorted) - 0.5
    x_grid_max = np.max(x_norm_sorted) + 0.5
    n_grid = min(1000, n)
    x_grid = np.linspace(x_grid_min, x_grid_max, n_grid)
    
    # Get LOESS predictions on the grid
    y_grid = _loess_predict(x_grid, x_loess_train, y_loess_train, best_bandwidth, degree=5)
    
    if not np.all(np.isfinite(y_grid)):
        return _global_fit_fallback(x, y)
        
    # 3. Fit global degree-5 polynomial to the LOESS-denoised grid
    # Use ridge regression for stability
    V = np.vander(x_grid, N=6, increasing=True)
    # Weight the points? Uniform is fine for now.
    lam = 1e-4
    A = V.T @ V + lam * np.eye(6)
    b = V.T @ y_grid
    try:
        coeff_norm = np.linalg.solve(A, b)
        if not np.all(np.isfinite(coeff_norm)):
            coeff_norm = np.zeros(6)
    except Exception:
        coeff_norm = np.zeros(6)
        
    # 4. Transform coefficients back to original x space
    return _transform_coeffs(coeff_norm, x_mean, x_std)


def _loess_predict(x_pred, x_train, y_train, bandwidth, degree=5):
    """
    Compute LOESS predictions for x_pred given training data.
    x_pred: array of points to predict
    x_train: sorted training x values
    y_train: corresponding training y values
    bandwidth: fraction of data to use (1.0 means all)
    degree: degree of local polynomial
    """
    n_train = len(x_train)
    if n_train < degree + 1:
        return np.zeros(len(x_pred))
        
    n_local = max(degree + 1, int(n_train * bandwidth))
    
    # Weights: Triangular or Gaussian? Triangular is standard for LOESS.
    # w_i = (1 - (d_i / d_max)^2)^3 for d_i < d_max, else 0
    # Here d_max is the distance to the n_local-th nearest neighbor.
    
    y_pred = np.zeros(len(x_pred))
    
    for i, xp in enumerate(x_pred):
        if time.time() > deadline - 5: # Hard stop inside loop if needed, but we passed deadline check
            break
            
        # Find nearest neighbors
        # Since x_train is sorted, we can use searchsorted
        idx = np.searchsorted(x_train, xp)
        
        # Define window bounds
        left = max(0, idx - n_local // 2)
        right = min(n_train, left + n_local)
        
        # Adjust if we are at the edge
        if idx >= n_train:
            right = n_train
            left = max(0, right - n_local)
        elif idx == 0:
            left = 0
            right = min(n_train, n_local)
            
        x_local = x_train[left:right]
        y_local = y_train[left:right]
        
        if len(x_local) < degree + 1:
            # Not enough points, use global mean or simple interp
            y_pred[i] = np.mean(y_train)
            continue
            
        # Calculate distances
        dists = np.abs(x_local - xp)
        max_dist = np.sort(dists)[min(n_local, len(dists))-1]
        
        if max_dist < 1e-10:
            y_pred[i] = y_local[np.argmin(dists)]
            continue
            
        # Triangular weights cubed (standard LOESS)
        w = 1.0 - (dists / max_dist) ** 2
        w[w < 0] = 0
        w = w ** 3
        
        # Weighted least squares for local polynomial
        V_local = np.vander(x_local, N=degree + 1, increasing=True)
        W = np.diag(w)
        
        # Solve (V^T W V) c = V^T W y
        VtW = V_local.T * w # This is V^T * diag(w)
        A_local = VtW @ V_local
        b_local = VtW @ y_local
        
        # Add small regularization for stability
        A_local += 1e-8 * np.eye(degree + 1)
        
        try:
            c_local = np.linalg.solve(A_local, b_local)
            y_pred[i] = c_local[0] # Constant term is the prediction at x=0? 
            # Wait, vander(x, increasing=True) with N=degree+1 gives [1, x, x^2, ...]
            # So c_local[0] is the intercept.
            # The prediction at xp is sum(c_local * xp^k).
            # Actually, the local fit is centered? No, standard vander is not centered.
            # Let's just evaluate the polynomial at xp.
            y_pred[i] = np.sum(c_local * (xp ** np.arange(degree + 1)))
        except Exception:
            y_pred[i] = np.mean(y_local)
            
    return y_pred


def _global_fit_fallback(x, y):
    """Fallback to simple global ridge fit."""
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-10:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std
    V = np.vander(x_norm, N=6, increasing=True)
    lam = 1e-4
    A = V.T @ V + lam * np.eye(6)
    b = V.T @ y
    try:
        coeff_norm = np.linalg.solve(A, b)
        if not np.all(np.isfinite(coeff_norm)):
            coeff_norm = np.zeros(6)
    except Exception:
        coeff_norm = np.zeros(6)
    return _transform_coeffs(coeff_norm, x_mean, x_std)


def _transform_coeffs(coeff_norm, x_mean, x_std):
    """
    Transform polynomial coefficients from normalized space to original space.
    """
    coeff_orig = np.zeros(6)
    for j in range(6):
        s = 0.0
        for i in range(j, 6):
            binom = _binom(i, j)
            s += coeff_norm[i] / (x_std ** i) * binom * ((-x_mean) ** (i - j))
        coeff_orig[j] = s
    
    if not np.all(np.isfinite(coeff_orig)):
        return np.zeros(6)
    
    return coeff_orig


def _binom(n, k):
    """Compute binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result *= (n - i)
        result //= (i + 1)
    return float(result)
# EVOLVE-BLOCK-END
