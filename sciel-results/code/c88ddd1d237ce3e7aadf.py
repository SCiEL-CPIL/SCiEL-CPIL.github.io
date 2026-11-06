"""Improved program for the suite task; uses weighted polynomial fit for better generalization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Returns 6 coefficients in ascending powers of x.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Ensure inputs are float64 numpy arrays
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Handle edge case of empty or tiny training sets
    if len(x) < 6:
        return np.zeros(6)
        
    # Check time
    if time.time() - start_time > deadline:
        return np.zeros(6)
        
    try:
        # Calculate weights based on Gaussian kernel centered at mean of x
        # This focuses the fit on the most probable states of the slow variable
        x_mean = np.mean(x)
        x_std = np.std(x)
        if x_std < 1e-8:
            x_std = 1.0
            
        # Use a width that covers a significant portion of the distribution
        # Sigma = 1.5 * std ensures weights decay smoothly but remain significant for most data
        sigma = 1.5 * x_std
        weights = np.exp(-0.5 * ((x - x_mean) / sigma) ** 2)
        
        # Ensure weights are not zero to avoid numerical issues in weighted least squares
        weights = np.maximum(weights, 1e-10)
        
        # Construct the Vandermonde matrix for degree 5 polynomial
        # Columns: [1, x, x^2, x^3, x^4, x^5]
        V = np.vander(x, 6, increasing=True)
        
        # Weighted least squares: min || W^(1/2) (V c - y) ||^2
        # Solution: c = (V^T W V)^-1 V^T W y
        # Using np.linalg.lstsq with weighted data is more stable
        sqrt_w = np.sqrt(weights)
        V_w = V * sqrt_w[:, np.newaxis]
        y_w = y * sqrt_w
        
        # Solve the weighted least squares problem
        coeffs, residuals, rank, s = np.linalg.lstsq(V_w, y_w, rcond=None)
        
        # Ensure coefficients are finite
        if not np.all(np.isfinite(coeffs)):
            # Fallback to unweighted fit if weighted fit is unstable
            coeffs_unw = np.polynomial.polynomial.polyfit(x, y, 5)
            if np.all(np.isfinite(coeffs_unw)):
                return coeffs_unw
            # Final fallback to linear
            coeffs_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            coeffs = np.zeros(6)
            coeffs[:2] = coeffs_lin
            return coeffs
            
        return coeffs
        
    except Exception:
        # Fallback to standard unweighted polynomial fit
        try:
            coeffs = np.polynomial.polynomial.polyfit(x, y, 5)
            if np.all(np.isfinite(coeffs)):
                return coeffs
        except Exception:
            pass
        
        # Final fallback to linear fit
        try:
            coeffs_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            coeffs = np.zeros(6)
            coeffs[:2] = coeffs_lin
            return coeffs
        except Exception:
            return np.zeros(6)
# EVOLVE-BLOCK-END
