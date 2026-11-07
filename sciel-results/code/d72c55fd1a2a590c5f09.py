import numpy as np
import time

# Global time tracking for safety
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # 5 second margin from 120s limit

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a sparse linear combination of Radial Basis Functions (RBFs).
    
    We use 10 RBF terms with fixed centers and widths. The centers are placed on a deterministic
    grid over [-1, 1] x [-1, 1] to ensure good coverage. The widths are fixed to provide
    localized yet overlapping support. The optimizer only fits the 10 amplitudes (params).
    
    This design decouples structural complexity (fixed geometry) from fitting (amplitudes),
    allowing the model to capture localized, non-polynomial features without the instability
    of optimizing geometric parameters.
    """
    
    # Safety check: If we are running out of time, return a simple stable prediction
    # Fallback to a simple linear combination of the first few RBFs or just zero if needed
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Use only the first 5 RBFs for speed in emergency
        centers = _get_rbf_centers()[:5]
        widths = _get_rbf_widths()[:5]
        pred = np.zeros_like(x)
        for i in range(5):
            dx = x - centers[i][0]
            dy = y - centers[i][1]
            r2 = dx*dx + dy*dy
            pred += params[i] * np.exp(-r2 / (2 * widths[i] * widths[i]))
        return pred

    # Get fixed RBF parameters
    centers = _get_rbf_centers()
    widths = _get_rbf_widths()
    
    pred = np.zeros_like(x)
    
    for i in range(10):
        cx, cy = centers[i]
        w = widths[i]
        dx = x - cx
        dy = y - cy
        r2 = dx*dx + dy*dy
        # Gaussian RBF: exp(-r^2 / (2 * w^2))
        # Using exp(-r2 / w2) for slightly tighter localization, or standard Gaussian.
        # Let's use standard Gaussian with sigma = w
        pred += params[i] * np.exp(-r2 / (2 * w * w))

    # Ensure numerical stability
    if not np.all(np.isfinite(pred)):
        # Fallback to simple polynomial if RBFs blew up (unlikely with fixed params)
        return (params[0] 
                + params[1] * x 
                + params[2] * y 
                + params[3] * x*x 
                + params[4] * y*y 
                + params[5] * x * y)

    return pred

def _get_rbf_centers():
    """
    Returns 10 fixed centers for RBFs.
    """
    return [
        (0.0, 0.0),
        (-0.7, -0.7),
        (0.7, -0.7),
        (-0.7, 0.7),
        (0.7, 0.7),
        (0.0, -0.8),
        (0.0, 0.8),
        (-0.8, 0.0),
        (0.8, 0.0),
        (0.3, 0.3)
    ]

def _get_rbf_widths():
    """
    Returns 10 fixed widths (sigma) for RBFs.
    We vary the widths to capture features at different scales.
    Larger widths for the center, smaller for the periphery to allow local adjustment.
    """
    return [
        1.0,   # Center, broad
        0.6,   # Corners, medium
        0.6,
        0.6,
        0.6,
        0.5,   # Edges, narrower
        0.5,
        0.5,
        0.5,
        0.4    # Detail point, narrow
    ]

