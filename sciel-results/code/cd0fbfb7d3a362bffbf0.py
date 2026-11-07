import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs (x, y) drawn from [-1, 1]^2.
    
    Strategy: Hybrid model combining a quadratic polynomial basis with two
    flexible tanh non-linear terms.
    
    The 10 parameters are assigned as follows:
    p[0]: Constant
    p[1]: Coefficient for x
    p[2]: Coefficient for y
    p[3]: Coefficient for x^2
    p[4]: Coefficient for y^2
    p[5]: Coefficient for x*y
    p[6]: x-weight for tanh term 1
    p[7]: y-weight for tanh term 1
    p[8]: x-weight for tanh term 2
    p[9]: y-weight for tanh term 2
    
    The model is:
    y_pred = p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*x*y 
             + tanh(p6*x + p7*y) + tanh(p8*x + p9*y)
             
    This structure retains the strong quadratic baseline (which handles many
    smooth functions well) while adding two adaptive non-linear components
    that can capture bumps, steps, or other non-polynomial features by
    optimizing their orientation and position (implicitly, via the weights,
    though biases are fixed at 0 for this configuration to save params).
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Ensure we have exactly 10 parameters
    if p.shape[0] != 10:
        p_new = np.ones(10, dtype=np.float64)
        n = min(p.shape[0], 10)
        p_new[:n] = p[:n]
        p = p_new

    # Polynomial part
    poly_part = (p[0] + 
                 p[1] * x + 
                 p[2] * y + 
                 p[3] * (x * x) + 
                 p[4] * (y * y) + 
                 p[5] * (x * y))
    
    # Non-linear part
    # Note: No biases in tanh inputs to save parameters.
    # The optimizer can shift the effective center by adjusting the 
    # interaction with the polynomial terms if needed, or the 
    # fixed zero-bias is sufficient for symmetric features.
    # However, missing biases might limit ability to center bumps away from 0.
    # But with only 10 params, this is a reasonable trade-off.
    
    h1 = np.tanh(p[6] * x + p[7] * y)
    h2 = np.tanh(p[8] * x + p[9] * y)
    
    # Output weights for tanh terms are fixed to 1.0.
    # This reduces parameter count but allows the tanh to contribute fully.
    # If the target feature is small, the polynomial part can compensate.
    # Alternatively, we could scale, but we have no params left.
    
    result = poly_part + h1 + h2
    
    return result
