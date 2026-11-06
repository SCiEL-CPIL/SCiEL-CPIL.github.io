"""Strategy: Nadaraya-Watson smoothing denoising followed by degree-5 polynomial fit with bandwidth grid search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n < 10:
        return np.zeros(6)
    
    # Remove any NaNs or Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    
    if n < 10:
        return np.zeros(6)

    # Strategy: Nadaraya-Watson kernel smoothing to estimate E[y|x], then fit polynomial.
    # This provides smooth denoising that adapts to local data density.
    
    x_min, x_max = np.min(x), np.max(x)
    if x_max == x_min:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.mean(y)])
    
    x_range = x_max - x_min
    
    # Sort x for efficient kernel evaluation
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Generate evaluation points for the smoothed curve
    n_eval = min(500, n)
    eval_x = np.linspace(x_min, x_max, n_eval)
    
    # Bandwidth grid search: try multiple bandwidths and select best via cross-validation
    # Silverman's rule of thumb as starting point
    sigma_x = np.std(x)
    if sigma_x == 0:
        sigma_x = x_range / 10.0
    silverman_h = 1.06 * sigma_x * n ** (-1.0/5.0)
    
    # Create bandwidth candidates around Silverman estimate
    # Use log-spaced grid from 0.3x to 3x Silverman
    h_grid = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]) * silverman_h
    h_grid = np.sort(np.unique(h_grid))
    
    # For each bandwidth, compute smoothed values and fit polynomial
    # Use a subset of data for speed if n is very large
    if n > 20000:
        # Subsample for bandwidth selection
        rng = np.random.RandomState(42)
        sub_idx = rng.choice(n, size=20000, replace=False)
        x_sub = x_sorted[sub_idx]
        y_sub = y_sorted[sub_idx]
        n_sub = len(x_sub)
    else:
        x_sub = x_sorted
        y_sub = y_sorted
        n_sub = n
    
    best_h = silverman_h
    best_score = np.inf
    best_smoothed_y = None
    
    # Evaluate each bandwidth
    for h in h_grid:
        if time.time() > deadline - 20.0:
            break
            
        # Compute Nadaraya-Watson smoothed values at eval points
        # Efficient: for each eval point, find nearby data points using searchsorted
        # Kernel: Gaussian K(u) = exp(-0.5*u^2)
        
        smoothed_y = np.zeros_like(eval_x)
        
        # For efficiency, process in chunks
        chunk_size = 100
        for i in range(0, len(eval_x), chunk_size):
            chunk = eval_x[i:i+chunk_size]
            
            # For each eval point, compute weighted average
            for j in range(len(chunk)):
                # Find points within ~3*h of eval point
                left = np.searchsorted(x_sub, chunk[j] - 3*h, side='left')
                right = np.searchsorted(x_sub, chunk[j] + 3*h, side='right')
                
                if right <= left:
                    smoothed_y[i+j] = y_sub[0] if n_sub > 0 else 0.0
                    continue
                    
                x_local = x_sub[left:right]
                y_local = y_sub[left:right]
                
                # Gaussian kernel weights
                u = (x_local - chunk[j]) / h
                weights = np.exp(-0.5 * u * u)
                
                w_sum = np.sum(weights)
                if w_sum > 1e-12:
                    smoothed_y[i+j] = np.sum(weights * y_local) / w_sum
                else:
                    smoothed_y[i+j] = np.mean(y_local)
        
        # Fit polynomial to smoothed curve
        try:
            coeff = np.polynomial.polynomial.polyfit(eval_x, smoothed_y, 5)
            if not np.all(np.isfinite(coeff)):
                continue
                
            # Cross-validation score: hold out 20% of data, evaluate MSE
            cv_frac = 0.2
            n_cv = max(100, int(n * cv_frac))
            cv_idx = np.random.RandomState(42).choice(n, size=n_cv, replace=False)
            
            # Compute polynomial predictions on held-out points
            x_cv = x[cv_idx]
            y_cv = y[cv_idx]
            y_pred = np.polynomial.polynomial.polyval(x_cv, coeff)
            
            cv_mse = np.mean((y_pred - y_cv) ** 2)
            
            if cv_mse < best_score:
                best_score = cv_mse
                best_h = h
                best_smoothed_y = smoothed_y.copy()
                
        except Exception:
            continue
    
    # If no valid bandwidth found, use Silverman
    if best_smoothed_y is None:
        h = silverman_h
        smoothed_y = np.zeros_like(eval_x)
        for j in range(len(eval_x)):
            left = np.searchsorted(x_sub, eval_x[j] - 3*h, side='left')
            right = np.searchsorted(x_sub, eval_x[j] + 3*h, side='right')
            if right <= left:
                smoothed_y[j] = y_sub[0] if n_sub > 0 else 0.0
            else:
                x_local = x_sub[left:right]
                y_local = y_sub[left:right]
                u = (x_local - eval_x[j]) / h
                weights = np.exp(-0.5 * u * u)
                w_sum = np.sum(weights)
                if w_sum > 1e-12:
                    smoothed_y[j] = np.sum(weights * y_local) / w_sum
                else:
                    smoothed_y[j] = np.mean(y_local)
        best_smoothed_y = smoothed_y
    
    # Final polynomial fit to the best smoothed curve
    try:
        if time.time() > deadline - 5.0:
            # Fallback to simple linear fit if running out of time
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            return np.pad(coeff, (0, 5-len(coeff)))
            
        coeff = np.polynomial.polynomial.polyfit(eval_x, best_smoothed_y, 5)
        
        if not np.all(np.isfinite(coeff)):
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            return np.pad(coeff, (0, 5-len(coeff)))
            
        return coeff

    except Exception:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            return np.pad(coeff, (0, 5-len(coeff)))
        except Exception:
            return np.zeros(6)
# EVOLVE-BLOCK-END
