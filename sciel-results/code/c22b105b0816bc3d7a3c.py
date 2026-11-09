"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial using kernel-weighted denoising.
    
    Strategy:
    1. Use Gaussian kernel weighting to estimate E[Y|X] at a set of evaluation points.
    2. Fit a degree-5 polynomial to these denoised pseudo-labels.
    3. Return coefficients in ascending powers of x.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin: finish by 100s, limit is 120s
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    if len(x) < 10:
        # Fallback: linear fit if too few points
        coeff = np.zeros(6)
        if len(x) >= 2:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff
    
    # Step 1: Estimate E[Y|X] using Gaussian kernel weighting at evaluation points
    # Choose evaluation points as quantiles to ensure good coverage
    n_eval = min(200, len(x))
    eval_points = np.quantile(x, np.linspace(0.01, 0.99, n_eval))
    
    # Gaussian kernel bandwidth - use MAD-based robust estimate
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-10:
        mad = np.std(x) + 1e-10
    bandwidth = 1.5 * mad * (n_eval / len(x)) ** 0.2
    
    # Compute kernel-weighted mean at each evaluation point
    denoised_y = np.zeros(n_eval)
    for i, xp in enumerate(eval_points):
        if time.time() > deadline - 5:
            # Use remaining points with simple interpolation
            break
        # Gaussian kernel weights
        z = (x - xp) / bandwidth
        w = np.exp(-0.5 * z**2)
        # Avoid division by zero
        w_sum = np.sum(w)
        if w_sum > 1e-10:
            denoised_y[i] = np.sum(w * y) / w_sum
        else:
            denoised_y[i] = y[np.argmin(np.abs(x - xp))]
    
    # Step 2: Fit degree-5 polynomial to denoised pseudo-labels
    # Use weighted least squares with kernel weights to emphasize reliable estimates
    try:
        # Fit polynomial to denoised data
        coeff = np.polynomial.polynomial.polyfit(eval_points, denoised_y, 5)
        
        # Verify coefficients are finite
        if not np.all(np.isfinite(coeff)):
            raise ValueError("Non-finite coefficients")
    except Exception:
        # Fallback: fit to raw data with regularization
        try:
            # Use numpy's polynomial fit with fallback to lower degree if needed
            coeff = np.zeros(6)
            for deg in [5, 4, 3, 2, 1]:
                try:
                    c = np.polynomial.polynomial.polyfit(x, y, deg)
                    if np.all(np.isfinite(c)):
                        coeff[:deg+1] = c
                        break
                except Exception:
                    continue
        except Exception:
            coeff = np.zeros(6)
    
    # Ensure all coefficients are finite
    coeff = np.where(np.isfinite(coeff), coeff, 0.0)
    
    return coeff
# EVOLVE-BLOCK-END
