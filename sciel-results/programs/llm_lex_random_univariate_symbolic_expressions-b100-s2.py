import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structural change: Fourier series (3 harmonics) + cubic polynomial.
    
    6 Fourier terms: cos(x), sin(x), cos(2x), sin(2x), cos(3x), sin(3x)
    4 polynomial terms: 1, x, x^2, x^3
    
    This is a fundamentally different basis family from Chebyshev+Gaussian RBFs.
    Fourier terms capture periodic/oscillatory structure; polynomials capture
    non-periodic trends. Together they form a rich 10-dimensional basis.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if params.shape[0] != 10:
        p = np.ones(10)
        n = min(10, params.shape[0])
        p[:n] = params[:n]
        params = p
    
    # Fourier basis: 3 harmonics (cos and sin at each)
    cos1 = np.cos(x)
    sin1 = np.sin(x)
    cos2 = np.cos(2.0 * x)
    sin2 = np.sin(2.0 * x)
    cos3 = np.cos(3.0 * x)
    sin3 = np.sin(3.0 * x)
    
    # Polynomial basis
    poly0 = np.ones_like(x)
    poly1 = x
    poly2 = x ** 2
    poly3 = x ** 3
    
    y = (params[0] * poly0 +
         params[1] * poly1 +
         params[2] * poly2 +
         params[3] * poly3 +
         params[4] * cos1 +
         params[5] * sin1 +
         params[6] * cos2 +
         params[7] * sin2 +
         params[8] * cos3 +
         params[9] * sin3)
    
    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)
    
    return y
