import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the linear combination of a fixed set of basis functions.
    
    Basis functions chosen to cover:
    - Polynomials in t=x/2 (1, t, t^2, t^3, t^4)
    - Trigonometric (sin(pi t), cos(pi t), sin(2 pi t))
    - Hyperbolic tangent (tanh(t)) for saturation
    - Scaled hyperbolic tangent (t * tanh(t))
    
    params: array of 10 coefficients.
    x: array of input points.
    """
    x = np.asarray(x, dtype=float)
    t = x / 2.0
    
    # Precompute basis functions
    # 1. Constant
    b0 = np.ones_like(t)
    
    # 2. Linear
    b1 = t.copy()
    
    # 3. Quadratic
    b2 = t ** 2
    
    # 4. Cubic
    b3 = t ** 3
    
    # 5. Quartic
    b4 = t ** 4
    
    # 6. Sin(pi t)
    b5 = np.sin(np.pi * t)
    
    # 7. Cos(pi t)
    b6 = np.cos(np.pi * t)
    
    # 8. Sin(2 pi t)
    b7 = np.sin(2.0 * np.pi * t)
    
    # 9. Tanh(t)
    b8 = np.tanh(t)
    
    # 10. t * Tanh(t)
    b9 = t * np.tanh(t)
    
    # Stack basis functions into a matrix (N, 10)
    # Rows are samples, columns are basis functions
    basis_matrix = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # Ensure params is a 1D array
    params = np.asarray(params, dtype=float).flatten()
    
    # If params has wrong length, pad or truncate (defensive)
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)), 'constant')
    else:
        params = params[:10]
        
    # Compute prediction
    pred = basis_matrix @ params
    
    return pred
