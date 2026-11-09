"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    x = x[:n]
    y = y[:n]
    
    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    # Strategy:
    # 1. Estimate the conditional expectation E[Y|X] using a non-parametric method 
    #    (Gaussian Kernel Regression with adaptive bandwidth).
    # 2. Fit a degree-5 polynomial to these "pseudo-labels" (x_grid, y_est).
    #    This separates the denoising step from the polynomial projection,
    #    allowing the polynomial to best approximate the smoothed underlying
    #    function rather than the noisy raw data.
    
    # Define grid for evaluating the conditional mean
    x_min, x_max = np.min(x), np.max(x)
    # Add buffer to cover potential extrapolation needs slightly
    buffer = 0.05 * (x_max - x_min) if x_max > x_min else 0.1
    x_min -= buffer
    x_max += buffer
    
    num_grid = 100
    x_grid = np.linspace(x_min, x_max, num_grid)
    
    # Estimate E[Y|X] at grid points using Gaussian Kernel Regression
    # Bandwidth selection: Use a simple heuristic based on data spread
    # Sigma = range / sqrt(n) * factor, or just a fixed fraction of range.
    # Let's try a few bandwidths and pick the one that minimizes a simple
    # cross-validation or just use a robust default.
    # For speed and robustness, we'll use a bandwidth that is roughly 10% of the range.
    sigma = (x_max - x_min) / 15.0
    
    # If sigma is too small, increase it to avoid under-smoothing
    if sigma < 1e-6:
        sigma = 0.1
    
    y_grid = np.zeros_like(x_grid)
    
    # Compute weights and means for each grid point
    # To make this faster, we can vectorize if possible, but n might be large.
    # Let's loop over grid points (100 is small) and vectorize over x, y.
    for i in range(num_grid):
        xi = x_grid[i]
        dist = x - xi
        # Weights: Gaussian kernel
        weights = np.exp(-0.5 * (dist / sigma) ** 2)
        w_sum = np.sum(weights)
        if w_sum > 1e-12:
            y_grid[i] = np.sum(weights * y) / w_sum
        else:
            # Fallback to global mean if no data nearby
            y_grid[i] = np.mean(y)
        
        # Time check
        if time.time() - t_start > deadline - 10.0:
            break

    # If we broke out early, fill remaining y_grid with global mean
    # (This is unlikely with 100 points)
    # Fit a degree-5 polynomial to the denoised curve (x_grid, y_grid)
    
    # Use least squares fit
    # We want c such that sum(c_i * x^i) approx y_grid
    # A = [1, x, x^2, x^3, x^4, x^5]
    A = np.vander(x_grid, 6, increasing=True)
    
    try:
        # Solve least squares
        coeff_asc, residuals, rank, sv = np.linalg.lstsq(A, y_grid, rcond=None)
        
        # Ensure all coefficients are finite
        if not np.all(np.isfinite(coeff_asc)):
             # Fallback to linear fit if numerical issues
             A_lin = np.vander(x_grid, 2, increasing=True)
             coeff_lin, _, _, _ = np.linalg.lstsq(A_lin, y_grid, rcond=None)
             coeff_asc = np.zeros(6)
             coeff_asc[:2] = coeff_lin
             
    except Exception:
        # Fallback to simple linear fit on raw data if everything fails
        try:
            A_lin = np.vander(x, 2, increasing=True)
            coeff_lin, _, _, _ = np.linalg.lstsq(A_lin, y, rcond=None)
            out = np.zeros(6)
            out[:2] = coeff_lin
            return out
        except Exception:
            return np.zeros(6, dtype=np.float64)
            
    # Optional: Refine coefficients using a small number of optimization steps
    # to minimize the error on the training data directly, but constrained
    # by the structure of the polynomial? No, that's just standard regression.
    # The key difference is fitting to the denoised y_grid.
    
    # However, the task asks to minimize normalized error on VALIDATION trajectories.
    # We don't have validation data. We assume that denoising the training data
    # provides a better estimate of the true function, which generalizes better.
    
    # One potential issue: if the true function is not well-approximated by a
    # degree-5 polynomial, this is the best we can do.
    
    # Let's try to improve the bandwidth selection slightly if time permits.
    # We can test a few bandwidths and pick the one that minimizes the
    # residual sum of squares of the polynomial fit to the denoised data?
    # No, that's circular.
    
    # Instead, let's just return the coefficients from the fit to the denoised data.
    
    return coeff_asc
# EVOLVE-BLOCK-END
