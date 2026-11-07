import numpy as np
import time

_DEADLINE = 140.0
_start_time = None

def _legendre_poly(n, x):
    """Compute Legendre polynomial P_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    P_prev = np.ones_like(x)
    P_curr = x.copy()
    for k in range(1, n):
        P_next = ((2 * k + 1) * x * P_curr - k * P_prev) / (k + 1)
        P_prev = P_curr
        P_curr = P_next
    return P_curr

def _build_features(x: np.ndarray) -> np.ndarray:
    """
    Build 10 basis functions:
    - 4 Legendre polynomials (orthogonal, well-conditioned)
    - 2 trigonometric (sin, cos) for periodic components
    - 2 tanh (for step-like/saturating features)
    - 2 localized Gaussian bumps (for local structure)
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    
    F = np.zeros((n, 10), dtype=np.float64)
    
    # --- Legendre polynomials (orthogonal, well-conditioned) ---
    # Scale x to [-1, 1] for proper Legendre polynomial behavior
    u = x / 3.0
    F[:, 0] = _legendre_poly(0, u)
    F[:, 1] = _legendre_poly(1, u)
    F[:, 2] = _legendre_poly(2, u)
    F[:, 3] = _legendre_poly(3, u)
    
    # --- Trigonometric ---
    F[:, 4] = np.sin(x)
    F[:, 5] = np.cos(x)
    
    # --- Tanh (saturating, good for step-like features) ---
    F[:, 6] = np.tanh(x)
    F[:, 7] = np.tanh(2.0 * x)
    
    # --- Localized Gaussians (for local bumps/structure) ---
    width = 1.0
    F[:, 8] = np.exp(-0.5 * ((x - (-1.5)) / width) ** 2)
    F[:, 9] = np.exp(-0.5 * ((x - 1.5) / width) ** 2)
    
    return F

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return np.zeros_like(np.asarray(x, dtype=np.float64))
    
    F = _build_features(x)
    result = F @ params
    
    # Ensure finite values
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
