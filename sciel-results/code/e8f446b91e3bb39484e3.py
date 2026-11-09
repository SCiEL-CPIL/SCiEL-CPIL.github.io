import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: 2 Gaussians + Cubic Polynomial.
    Params:
    0: a1 (Gaussian 1 amplitude)
    1: c1 (Gaussian 1 center)
    2: log_w1 (log of Gaussian 1 width)
    3: a2 (Gaussian 2 amplitude)
    4: c2 (Gaussian 2 center)
    5: log_w2 (log of Gaussian 2 width)
    6: b0 (Poly constant)
    7: b1 (Poly linear)
    8: b2 (Poly quadratic)
    9: b3 (Poly cubic)
    """
    # Unpack parameters
    a1 = params[0]
    c1 = params[1]
    log_w1 = params[2]
    a2 = params[3]
    c2 = params[4]
    log_w2 = params[5]
    b0 = params[6]
    b1 = params[7]
    b2 = params[8]
    b3 = params[9]
    
    # Ensure widths are positive and reasonable
    # Using exp ensures positivity. 
    # If log_w is large, width is large (broad bump).
    # If log_w is small, width is small (sharp bump).
    w1 = np.exp(log_w1)
    w2 = np.exp(log_w2)
    
    # Calculate Gaussians
    # Clip the exponent to avoid overflow/underflow issues in exp
    # (x - c)^2 / (2 * w^2)
    # If w is very small, this term can be huge.
    # Let's clamp the argument to exp.
    
    g1_arg = -0.5 * ((x - c1) / w1) ** 2
    g2_arg = -0.5 * ((x - c2) / w2) ** 2
    
    # Clip to [-50, 50] to keep exp stable
    g1_arg = np.clip(g1_arg, -50, 50)
    g2_arg = np.clip(g2_arg, -50, 50)
    
    g1 = a1 * np.exp(g1_arg)
    g2 = a2 * np.exp(g2_arg)
    
    # Calculate Polynomial
    poly = b0 + b1 * x + b2 * x**2 + b3 * x**3
    
    # Sum
    y = g1 + g2 + poly
    
    return y
