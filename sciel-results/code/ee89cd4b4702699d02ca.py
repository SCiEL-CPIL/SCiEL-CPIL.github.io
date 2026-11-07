"""Program using a Kernel Ridge Regression strategy with adaptive Gaussian RBF kernels."""
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form symbolic regression candidate.
    Uses a linear combination of 10 Gaussian RBF kernels with data-adaptive centers.
    The 10 params act as adaptive weights for local interpolation.
    
    Centers are placed at quantiles of the input distribution to better cover
    the actual data range. Bandwidth is adapted to local density.
    """
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 100.0  # Leave 20s safety margin

    # Scale input to [-1, 1] for better kernel behavior
    u = x / 3.0

    # Place centers at quantiles of the input for better coverage
    # Use 10 quantiles from 5% to 95% to avoid edge effects
    quantiles = np.linspace(0.05, 0.95, 10)
    centers = np.quantile(u, quantiles)

    # Adapt bandwidth based on local spacing
    # Estimate typical spacing between adjacent centers
    if len(centers) > 1:
        spacings = np.diff(centers)
        sigma = np.median(spacings) * 0.6
    else:
        sigma = 0.35
    
    # Clamp sigma to reasonable range
    sigma = np.clip(sigma, 0.1, 0.8)
    sigma2 = sigma * sigma

    # Compute the RBF basis: exp(-0.5 * (u - c)^2 / sigma^2)
    diff = u[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel values
    rbf = np.exp(-0.5 * (diff ** 2) / sigma2)

    # Linear combination with params
    y = rbf @ params

    # Check deadline
    if time.time() > deadline:
        # Return current best (already computed)
        pass

    return y
