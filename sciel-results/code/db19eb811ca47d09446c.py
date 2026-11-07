import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 1 input.
    
    Strategy: Linear model with a bilinear Chebyshev product basis.
    This uses 10 coefficients for a basis of T_i(x) * T_j(x) for selected pairs,
    which allows capturing sharp non-polynomial features more efficiently than
    a single linear basis. The model is linear in params, so BFGS converges
    to the global minimum efficiently.
    
    Basis functions (all in x scaled to [-1,1]):
    0: T0*T0 = 1
    1: T1*T0 = x
    2: T2*T0 = 2x^2 - 1
    3: T3*T0 = 4x^3 - 3x
    4: T2*T1 = (2x^2-1)*x = 2x^3 - x
    5: T3*T1 = (4x^3-3x)*x = 4x^4 - 3x^2
    6: T2*T2 = (2x^2-1)^2 = 4x^4 - 4x^2 + 1
    7: T3*T2 = (4x^3-3x)(2x^2-1) = 8x^5 - 10x^3 + 3x
    8: T3*T3 = (4x^3-3x)^2 = 16x^6 - 24x^4 + 9x^2
    9: T4*T0 = 8x^4 - 8x^2 + 1
    """
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
    
    try:
        # Scale x to [-1, 1]
        xs = x / 2.0
        
        # Compute Chebyshev polynomials up to T4
        T0 = np.ones_like(xs)
        T1 = xs
        T2 = 2.0 * xs**2 - 1.0
        T3 = 4.0 * xs**3 - 3.0 * xs
        T4 = 8.0 * xs**4 - 8.0 * xs**2 + 1.0
        
        # Bilinear basis: T_i * T_j for selected pairs
        basis = np.column_stack([
            T0 * T0,  # 0: 1
            T1 * T0,  # 1: x
            T2 * T0,  # 2: 2x^2 - 1
            T3 * T0,  # 3: 4x^3 - 3x
            T2 * T1,  # 4: 2x^3 - x
            T3 * T1,  # 5: 4x^4 - 3x^2
            T2 * T2,  # 6: 4x^4 - 4x^2 + 1
            T3 * T2,  # 7: 8x^5 - 10x^3 + 3x
            T3 * T3,  # 8: 16x^6 - 24x^4 + 9x^2
            T4 * T0,  # 9: 8x^4 - 8x^2 + 1
        ])
        
        result = basis @ params
        
        return result
        
    except Exception:
        return np.zeros_like(x)
