"""Rewritten program using Iteratively Reweighted Least Squares (IRLS) for robust fitting."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the subgrid tendency as a function of x.
    
    Uses Iteratively Reweighted Least Squares (IRLS) with a Huber loss function
    to robustly fit the polynomial coefficients, down-weighting outliers.
    """
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    # Filter out any non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    n = len(x)
    if n < 6:
        return np.zeros(6)
    
    # Build the raw Vandermonde matrix
    V = np.vander(x, N=6, increasing=True)
    
    # Initial weights are all 1
    weights = np.ones(n)
    
    # Huber loss parameter 'delta' controls the threshold for outlier penalty.
    # We estimate a robust scale (e.g., MAD) to set delta.
    # For IRLS, we typically use weights w_i = 1 if |r_i| <= delta, else delta/|r_i|
    
    # Estimate scale of residuals
    # Use median absolute deviation (MAD) of y as a proxy for scale
    mad = np.median(np.abs(y - np.median(y)))
    scale = 1.4826 * mad if mad > 0 else 1.0
    delta = 1.5 * scale # Standard Huber delta
    
    c = np.zeros(6)
    
    for _ in range(50): # Max iterations
        # Weighted Least Squares
        # Minimize sum( w_i * (y_i - V_i . c)^2 )
        # (V^T W V) c = V^T W y
        W = np.diag(weights)
        # To avoid constructing large diagonal matrix, we can do:
        # G = V.T @ (weights[:, None] * V)
        # b = V.T @ (weights * y)
        
        weighted_V = V * np.sqrt(weights)[:, None]
        weighted_y = y * np.sqrt(weights)
        
        try:
            c, residuals, rank, s = np.linalg.lstsq(weighted_V, weighted_y, rcond=None)
        except np.linalg.LinAlgError:
            c = np.zeros(6)
            break
        
        # Compute residuals
        y_pred = V @ c
        residuals = y - y_pred
        
        # Update weights based on Huber loss
        # w_i = 1 if |r_i| <= delta, else delta / |r_i|
        abs_res = np.abs(residuals)
        new_weights = np.where(abs_res <= delta, 1.0, delta / (abs_res + 1e-12))
        
        # Check convergence
        if np.allclose(weights, new_weights, rtol=1e-5, atol=1e-8):
            weights = new_weights
            break
        
        weights = new_weights
    
    # Final sanity check
    if not np.all(np.isfinite(c)):
        c = np.zeros(6)
    
    return c
# EVOLVE-BLOCK-END
