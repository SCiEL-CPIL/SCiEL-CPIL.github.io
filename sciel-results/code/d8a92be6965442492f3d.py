import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Quadratic polynomial + Fourier series (4 harmonics).
    
    3 Polynomial terms: 1, x, x^2
    7 Fourier terms: cos(x), sin(x), cos(2x), sin(2x), cos(3x), sin(3x), cos(4x)
    
    This basis combines global polynomial trends with periodic components.
    The Fourier terms are bounded and well-conditioned, which helps BFGS
    optimization starting from ones. The polynomial terms capture non-periodic
    drift and curvature.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure params has length 10
    if params.shape[0] != 10:
        p = np.ones(10)
        n = min(10, params.shape[0])
        p[:n] = params[:n]
        params = p
    
    # Polynomial basis
    p0 = np.ones_like(x)
    p1 = x
    p2 = x ** 2
    
    # Fourier basis
    c1 = np.cos(x)
    s1 = np.sin(x)
    c2 = np.cos(2.0 * x)
    s2 = np.sin(2.0 * x)
    c3 = np.cos(3.0 * x)
    s3 = np.sin(3.0 * x)
    c4 = np.cos(4.0 * x)
    
    y = (params[0] * p0 +
         params[1] * p1 +
         params[2] * p2 +
         params[3] * c1 +
         params[4] * s1 +
         params[5] * c2 +
         params[6] * s2 +
         params[7] * c3 +
         params[8] * s3 +
         params[9] * c4)
    
    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)
    
    return y
