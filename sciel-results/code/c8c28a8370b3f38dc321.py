import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Product-based feature expansion: linear combination of 10 terms.
    Each term is a low-degree Chebyshev polynomial times a nonlinear activation.
    
    params: array of length 10
    - params[0..2]: coefficients for Chebyshev T0, T1, T2 (degree 0,1,2) with tanh(x) activation
    - params[3..4]: coefficients for Chebyshev T0, T1 with sin(x) activation
    - params[5..6]: coefficients for Chebyshev T0, T1 with x^2 * sign(x) (smooth-ish cubic-like)
    - params[7..9]: coefficients for Chebyshev T0, T1, T2 with exp(-x^2) (Gaussian-like bump)
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Chebyshev polynomials of the first kind
    T0 = np.ones_like(x)
    T1 = x
    T2 = 2.0 * x * x - 1.0
    
    # Nonlinear activations
    act_tanh = np.tanh(x)
    act_sin = np.sin(x)
    act_cubic = x * np.abs(x)  # smooth cubic-like, odd function
    act_gauss = np.exp(-x * x)
    
    # Build the linear combination
    y = (params[0] * T0 + params[1] * T1 + params[2] * T2) * act_tanh
    y = y + (params[3] * T0 + params[4] * T1) * act_sin
    y = y + (params[5] * T0 + params[6] * T1) * act_cubic
    y = y + (params[7] * T0 + params[8] * T1 + params[9] * T2) * act_gauss
    
    return y
