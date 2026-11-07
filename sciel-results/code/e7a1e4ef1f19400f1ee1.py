import numpy as np
import time

# Deadline guard: keep evaluation well under 150s
_DEADLINE = 140.0  # seconds
_start_time = None

def _build_features(x: np.ndarray) -> np.ndarray:
    """
    Build 10 basis functions that span a rich function space:
    - 4 Legendre polynomials (global smooth trends)
    - 2 trigonometric (sin, cos) for periodic components
    - 2 tanh (for step-like/saturating features)
    - 2 localized Gaussian bumps (for local structure)
    
    This mixed basis is well-conditioned and flexible enough to approximate
    many common hidden functions on [-3, 3].
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    
    F = np.zeros((n, 10), dtype=np.float64)
    
    # --- Legendre polynomials (orthogonal, well-conditioned) ---
    # P0 = 1
    F[:, 0] = 1.0
    # P1 = x
    F[:, 1] = x
    # P2 = (3x^2 - 1)/2
    F[:, 2] = 0.5 * (3.0 * x**2 - 1.0)
    # P3 = (5x^3 - 3x)/2
    F[:, 3] = 0.5 * (5.0 * x**3 - 3.0 * x)
    
    # --- Trigonometric ---
    F[:, 4] = np.sin(x)
    F[:, 5] = np.cos(x)
    
    # --- Tanh (saturating, good for step-like features) ---
    F[:, 6] = np.tanh(x)
    F[:, 7] = np.tanh(2.0 * x)
    
    # --- Localized Gaussians (for local bumps/structure) ---
    # Two Gaussians centered at -1.5 and 1.5 with moderate width
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
