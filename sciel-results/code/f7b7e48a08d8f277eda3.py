"""Rewritten program: piecewise cubic spline closure for Lorenz-96 subgrid tendency."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import UnivariateSpline
import numpy.polynomial.polynomial as P


def construct(training_x, training_subgrid_tendency):
    """
    Fit a smoothing cubic spline to the training data, then approximate the
    resulting spline by a degree-5 polynomial via least squares on a dense grid.
    Returns 6 coefficients in ascending powers of x.
    
    Uses scipy.interpolate.UnivariateSpline for robust smoothing, which is
    generally more effective than binned means for regression tasks.
    """
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    n = len(x)
    if n < 10:
        # Fallback to linear fit if too few points
        coeff = np.zeros(6)
        if n >= 2:
            coeff[:2] = P.polyfit(x, y, 1)
        return coeff

    # Estimate noise level for smoothing parameter
    # Using Median Absolute Deviation (MAD) for robust scale estimation
    y_median = np.median(y)
    mad = np.median(np.abs(y - y_median))
    sigma_est = 1.4826 * mad
    if sigma_est == 0 or not np.isfinite(sigma_est):
        sigma_est = np.std(y)
        if sigma_est == 0 or not np.isfinite(sigma_est):
            sigma_est = 1.0 # Avoid division by zero or NaN

    # Smoothing parameter s
    # s = n * sigma^2 is the standard choice for white noise.
    # We might multiply by a factor to ensure sufficient smoothing for the polynomial fit.
    # A factor of 1.0 is a safe start.
    s = n * (sigma_est ** 2) * 1.0
    
    try:
        # Fit a smoothing spline
        # k=3 for cubic spline
        spl = UnivariateSpline(x, y, s=s, k=3)
        
        # Determine evaluation range
        x_min, x_max = np.min(x), np.max(x)
        if x_max == x_min:
            x_min -= 1.0
            x_max += 1.0
        margin = 0.05 * (x_max - x_min)
        
        # Evaluate on a dense grid
        n_eval = 1000
        x_eval = np.linspace(x_min - margin, x_max + margin, n_eval)
        y_eval = spl(x_eval)
        
        # Fit a degree-5 polynomial to the spline values
        coeff = P.polyfit(x_eval, y_eval, 5)
        
    except Exception:
        # Fallback to direct polynomial fit if spline fails
        try:
            coeff = P.polyfit(x, y, 5)
        except Exception:
            coeff = np.zeros(6)
            if n >= 2:
                coeff[:2] = P.polyfit(x, y, 1)

    # Ensure coefficients are finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)
        if n >= 2:
            coeff[:2] = P.polyfit(x, y, 1)
            
    return coeff
# EVOLVE-BLOCK-END
