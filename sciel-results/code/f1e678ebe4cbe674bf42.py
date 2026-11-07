import numpy as np
import time
import scipy.optimize as opt

_START_TIME = time.time()
_DEADLINE = 140.0  # seconds, keep well under 150s

def _build_features(x: np.ndarray) -> np.ndarray:
    """
    Build a 10-dimensional feature matrix combining:
    - Low-order Chebyshev polynomials (for smooth global structure)
    - Localized Gaussian bumps (for local flexibility)
    - Trigonometric terms (for periodic components)
    
    Features are designed to be well-conditioned for BFGS optimization.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    F = np.zeros((n, 10), dtype=np.float64)
    
    # Scale to [-1, 1] for Chebyshev polynomials
    u = x / 3.0
    
    # Features 0-3: Chebyshev polynomials T0 to T3
    # T0(x) = 1
    F[:, 0] = 1.0
    # T1(x) = x
    F[:, 1] = u
    # T2(x) = 2x^2 - 1
    F[:, 2] = 2.0 * u**2 - 1.0
    # T3(x) = 4x^3 - 3x
    F[:, 3] = 4.0 * u**3 - 3.0 * u
    
    # Features 4-7: Localized Gaussian bumps at strategic centers
    # Centers chosen to cover the domain with good overlap
    centers = np.array([-1.5, -0.5, 0.5, 1.5])
    width = 1.2  # Fixed width for all Gaussians
    for i, c in enumerate(centers):
        F[:, 4 + i] = np.exp(-0.5 * ((x - c) / width) ** 2)
    
    # Features 8-9: Trigonometric terms for periodic components
    F[:, 8] = np.sin(x)
    F[:, 9] = np.cos(x)
    
    return F

def _objective(params, F, y):
    """Compute MSE between prediction and target."""
    pred = F @ params
    if not np.all(np.isfinite(pred)):
        return 1e30
    return np.mean((pred - y) ** 2)

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using the 10-parameter linear combination
    of the predefined basis functions.
    
    The evaluator will fit params via BFGS; this function just
    evaluates the linear model.
    """
    F = _build_features(x)
    pred = F @ params
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(x)
    return pred
