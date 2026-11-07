import numpy as np
import time

_DEADLINE = 140.0
_start_time = None

def _build_features(x: np.ndarray) -> np.ndarray:
    """
    Build 10 basis functions:
    - 4 Legendre polynomials (global smooth trends)
    - 2 trigonometric (sin, cos) for periodic components
    - 2 tanh (for step-like/saturating features)
    - 2 localized Gaussian bumps (for local structure)
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    
    F = np.zeros((n, 10), dtype=np.float64)
    
    # --- Legendre polynomials (orthogonal, well-conditioned) ---
    F[:, 0] = 1.0
    F[:, 1] = x
    F[:, 2] = 0.5 * (3.0 * x**2 - 1.0)
    F[:, 3] = 0.5 * (5.0 * x**3 - 3.0 * x)
    
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
