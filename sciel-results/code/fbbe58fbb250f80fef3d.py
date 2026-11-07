import numpy as np

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
    x = np.asarray(x, dtype=np.float64)
    # Scale x to [-1, 1] for Legendre polynomials
    u = x / 3.0
    n = x.shape[0]
    F = np.zeros((n, 10), dtype=np.float64)
    
    # 1-6: Legendre Polynomials P0 to P5
    F[:, 0] = _legendre_poly(0, u)
    F[:, 1] = _legendre_poly(1, u)
    F[:, 2] = _legendre_poly(2, u)
    F[:, 3] = _legendre_poly(3, u)
    F[:, 4] = _legendre_poly(4, u)
    F[:, 5] = _legendre_poly(5, u)
    
    # 7-10: Trig and Gaussian
    F[:, 6] = np.sin(x)
    F[:, 7] = np.cos(x)
    F[:, 8] = np.sin(2.0 * x)
    F[:, 9] = np.exp(-0.5 * (x / 1.5) ** 2) # Gaussian at 0, width 1.5
    
    return F

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    F = _build_features(x)
    return F @ params
