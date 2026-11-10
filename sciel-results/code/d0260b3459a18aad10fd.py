"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved subgrid tendency
    using a weighted local polynomial regression approach averaged over the domain.
    """
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n < 6:
        # Fallback to simple linear if not enough data
        if n >= 2:
            c = np.zeros(6)
            c[:2] = np.polyfit(x, y, 1)[::-1]
            return c
        return np.zeros(6)
    
    # Sort data for efficient windowing
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Determine window size: aim for ~100-200 points per local fit
    # If n is large, use a fixed fraction; if small, use all
    window_size = max(20, min(200, n // 10))
    
    # We will compute local coefficients at a set of evaluation points
    # and then average them to get the global coefficients.
    # To ensure coverage and stability, we evaluate at the quantiles of x.
    # Let's use 50 evaluation points spread across the range.
    n_eval_points = 50
    eval_points = np.quantile(x_sorted, np.linspace(0.01, 0.99, n_eval_points))
    
    local_coeffs = np.zeros((n_eval_points, 6))
    
    # Precompute basis for local fits? No, x changes per window.
    # However, we can optimize by only fitting if we have time.
    
    for i, center in enumerate(eval_points):
        if time.time() > deadline - 5.0:
            break
            
        # Find indices in x_sorted within window_size of center
        # Use searchsorted for efficiency
        left_idx = np.searchsorted(x_sorted, center - window_size * (x_sorted[-1] - x_sorted[0]) / (2 * n) if n > 0 else 0)
        # Better window definition: fixed number of points or fixed range?
        # Fixed range is better for heteroscedasticity, but fixed points is faster.
        # Let's use a fixed range based on local density or global span.
        # Let's stick to a fixed number of points in the neighborhood for simplicity and speed,
        # but weighted by distance.
        
        # Actually, let's use a bandwidth based on the local spacing.
        # Find the index of the center
        idx_center = np.searchsorted(x_sorted, center)
        
        # Take points within a bandwidth h
        # Estimate h as 10 times the median local spacing
        if n > 10:
            spacings = np.diff(x_sorted)
            median_spacing = np.median(spacings)
            h = 10 * median_spacing
        else:
            h = (x_sorted[-1] - x_sorted[0]) / 2.0
        
        # Find points in [center - h, center + h]
        left = np.searchsorted(x_sorted, center - h)
        right = np.searchsorted(x_sorted, center + h)
        
        # Ensure we have at least 6 points
        if right - left < 6:
            # Expand window
            left = max(0, left - 10)
            right = min(n, right + 10)
        
        if right - left < 6:
            # Still not enough, use all data
            left = 0
            right = n
            
        x_win = x_sorted[left:right]
        y_win = y_sorted[left:right]
        
        # Compute weights: Gaussian kernel
        # w = exp(-0.5 * ((x_win - center) / (h / 2)) ** 2)
        w = np.exp(-0.5 * ((x_win - center) / (h / 2.0)) ** 2)
        
        # Weighted least squares for degree 5 polynomial
        V = np.vander(x_win, N=6, increasing=True)
        # Weight the matrix and vector
        V_w = V * np.sqrt(w)[:, np.newaxis]
        y_w = y_win * np.sqrt(w)
        
        # Solve (V_w^T V_w) c = V_w^T y_w
        # Add small ridge for stability
        A = V_w.T @ V_w + 1e-8 * np.eye(6)
        b = V_w.T @ y_w
        
        try:
            c_local = np.linalg.solve(A, b)
            local_coeffs[i] = c_local
        except np.linalg.LinAlgError:
            # Fallback to lstsq
            try:
                c_local = np.linalg.lstsq(A, b, rcond=None)[0]
                local_coeffs[i] = c_local
            except:
                local_coeffs[i] = 0.0
        
    # Average the local coefficients to get the global shared polynomial
    # We should only average over successful fits
    valid_mask = np.all(np.isfinite(local_coeffs), axis=1)
    if np.sum(valid_mask) > 0:
        global_coeffs = np.mean(local_coeffs[valid_mask], axis=0)
    else:
        # Fallback to global ridge if local fits failed
        V = np.vander(x, N=6, increasing=True)
        A = V.T @ V + 1e-4 * np.eye(6)
        b = V.T @ y
        try:
            global_coeffs = np.linalg.solve(A, b)
        except:
            global_coeffs = np.zeros(6)
            if n >= 2:
                global_coeffs[:2] = np.polyfit(x, y, 1)[::-1]
    
    # Ensure finite
    if not np.all(np.isfinite(global_coeffs)):
        global_coeffs = np.zeros(6)
        if n >= 2:
            global_coeffs[:2] = np.polyfit(x, y, 1)[::-1]
    
    return global_coeffs
# EVOLVE-BLOCK-END
