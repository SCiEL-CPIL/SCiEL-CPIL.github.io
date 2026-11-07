import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a 10-parameter Adaptive Composite Basis.
    
    The model is a linear combination of:
    - Polynomial terms: 1, x, x^2, x^3
    - Trigonometric terms: sin(x), cos(x), sin(2x), cos(2x)
    - Adaptive Hyperbolic tangent term: c * tanh(s * x)
    
    The last two parameters control the coefficient and scale of the
    tanh term, allowing the model to adapt the width and amplitude of
    sigmoidal transitions, while the rest remain a robust linear basis.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure params has exactly 10 elements
    if params.shape[0] != 10:
        return np.zeros_like(x)
        
    # Extract parameters
    c_poly = params[:4]       # Coeffs for 1, x, x^2, x^3
    c_trig = params[4:8]      # Coeffs for sin(x), cos(x), sin(2x), cos(2x)
    c_tanh = params[8]        # Coeff for tanh(s*x)
    s_tanh = params[9]        # Scale for tanh(s*x)
    
    # Construct basis terms
    # 1. Polynomial basis
    p1 = np.ones_like(x)
    p2 = x
    p3 = x**2
    p4 = x**3
    
    # 2. Trigonometric basis (fixed frequencies)
    t1 = np.sin(x)
    t2 = np.cos(x)
    t3 = np.sin(2 * x)
    t4 = np.cos(2 * x)
    
    # 3. Adaptive Hyperbolic tangent
    # Clamp scale to avoid extreme values that might cause numerical issues
    # though tanh is bounded, very large s leads to sharp steps which
    # can create steep gradients.
    s_safe = np.clip(s_tanh, 0.1, 10.0)
    h1 = np.tanh(s_safe * x)
    
    # Combine linearly
    # We can compute this directly without building a full matrix for efficiency
    # prediction = c0*p1 + c1*p2 + c2*p3 + c3*p4 + c4*t1 + c5*t2 + c6*t3 + c7*t4 + c8*h1
    
    prediction = (
        c_poly[0] * p1 +
        c_poly[1] * p2 +
        c_poly[2] * p3 +
        c_poly[3] * p4 +
        c_trig[0] * t1 +
        c_trig[1] * t2 +
        c_trig[2] * t3 +
        c_trig[3] * t4 +
        c_tanh * h1
    )
    
    # Ensure finite output
    if not np.all(np.isfinite(prediction)):
        return np.zeros_like(x)
        
    return prediction
