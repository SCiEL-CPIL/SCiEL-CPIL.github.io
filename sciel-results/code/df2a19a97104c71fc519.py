# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    # Handle edge cases
    if len(x) == 0:
        return np.zeros(6)
    
    # If time is already tight, fall back to simple polyfit
    if time.time() > deadline - 5:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            return coeff
        except Exception:
            return np.zeros(6)

    # Subsampling for GP if N is large
    max_points = 1000
    if len(x) > max_points:
        indices = np.random.choice(len(x), max_points, replace=False)
        x_gp = x[indices]
        y_gp = y[indices]
    else:
        x_gp = x
        y_gp = y
        
    # RBF Kernel parameters
    # Estimate length scale from data
    # Use median distance for robustness
    if len(x_gp) > 1:
        # Random pairs for distance estimation to save time
        n_pairs = min(1000, len(x_gp) * (len(x_gp) - 1) // 2)
        i1 = np.random.randint(0, len(x_gp), n_pairs)
        i2 = np.random.randint(0, len(x_gp), n_pairs)
        # Ensure i1 != i2
        mask = i1 != i2
        i1 = i1[mask]
        i2 = i2[mask]
        if len(i1) > 0:
            dists = np.abs(x_gp[i1] - x_gp[i2])
            sigma = np.median(dists)
            if sigma < 1e-8:
                sigma = 1.0
        else:
            sigma = 1.0
    else:
        sigma = 1.0
        
    gamma = 1.0 / (2.0 * sigma * sigma)
    
    # Noise variance
    var_y = np.var(y_gp)
    if var_y < 1e-12:
        var_y = 1.0
    sigma_n_sq = 1e-3 * var_y
    
    # Build Kernel Matrix
    # K_ij = exp(-gamma * (x_i - x_j)^2)
    x_diff = x_gp[:, np.newaxis] - x_gp[np.newaxis, :]
    K = np.exp(-gamma * x_diff * x_diff)
    K += sigma_n_sq * np.eye(len(x_gp))
    
    # Solve for alpha
    try:
        # Use Cholesky or solve
        alpha = np.linalg.solve(K, y_gp)
    except np.linalg.LinAlgError:
        # Add jitter if singular
        K += 1e-6 * np.eye(len(x_gp))
        try:
            alpha = np.linalg.solve(K, y_gp)
        except Exception:
            # Fallback to polyfit on original data
            try:
                coeff = np.polynomial.polynomial.polyfit(x, y, 5)
                return coeff
            except Exception:
                return np.zeros(6)

    # Generate grid for pseudo-labels
    # Cover the range of x with some padding
    x_min, x_max = np.min(x), np.max(x)
    # Expand range slightly to ensure coverage
    pad = 0.1 * (x_max - x_min) if x_max > x_min else 1.0
    x_min -= pad
    x_max += pad
    
    n_grid = 200
    x_grid = np.linspace(x_min, x_max, n_grid)
    
    # Predict on grid
    # K_star = exp(-gamma * (x_grid_i - x_gp_j)^2)
    x_diff_grid = x_grid[:, np.newaxis] - x_gp[np.newaxis, :]
    K_star = np.exp(-gamma * x_diff_grid * x_diff_grid)
    
    y_pred = K_star @ alpha
    
    # Check time again
    if time.time() > deadline - 10:
        try:
            coeff = np.polynomial.polynomial.polyfit(x_grid, y_pred, 5)
            return coeff
        except Exception:
            return np.zeros(6)

    # Fit degree-5 polynomial to pseudo-labels
    try:
        coeff = np.polynomial.polynomial.polyfit(x_grid, y_pred, 5)
        return coeff
    except Exception:
        # Fallback
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            return coeff
        except Exception:
            return np.zeros(6)

# EVOLVE-BLOCK-END
