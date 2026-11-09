import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: 2 Gaussians (bounded centers/widths) + Cubic Polynomial + small sinusoidal term.
    Adds mild smooth non-polynomial curvature (sin/cos) to better capture wiggles while
    keeping the problem smooth for BFGS.
    
    Params:
    0: a1 (Gaussian 1 amplitude)
    1: c1_raw (raw center for Gaussian 1, squashed to [-2, 2])
    2: log_w1 (log of Gaussian 1 width)
    3: a2 (Gaussian 2 amplitude)
    4: c2_raw (raw center for Gaussian 2, squashed to [-2, 2])
    5: log_w2 (log of Gaussian 2 width)
    6: b0 (Poly constant)
    7: b1 (Poly linear)
    8: b2 (Poly quadratic)
    9: b3 (Poly cubic)
    """
    # Unpack parameters
    a1 = params[0]
    c1_raw = params[1]
    log_w1 = params[2]
    a2 = params[3]
    c2_raw = params[4]
    log_w2 = params[5]
    b0 = params[6]
    b1 = params[7]
    b2 = params[8]
    b3 = params[9]
    
    # Squash centers to [-2, 2]
    c1 = 2.0 * np.tanh(c1_raw)
    c2 = 2.0 * np.tanh(c2_raw)
    
    # Widths: clip log_w to prevent extreme values
    log_w1 = np.clip(log_w1, -5, 5)
    log_w2 = np.clip(log_w2, -5, 5)
    w1 = np.exp(log_w1)
    w2 = np.exp(log_w2)
    
    # Calculate Gaussians with stable argument clipping
    g1_arg = -0.5 * ((x - c1) / w1) ** 2
    g2_arg = -0.5 * ((x - c2) / w2) ** 2
    g1_arg = np.clip(g1_arg, -50, 50)
    g2_arg = np.clip(g2_arg, -50, 50)
    
    g1 = a1 * np.exp(g1_arg)
    g2 = a2 * np.exp(g2_arg)
    
    # Polynomial part
    poly = b0 + b1 * x + b2 * x**2 + b3 * x**3
    
    # Small sinusoidal term to capture low-frequency wiggles without adding new parameters.
    # The amplitude is kept small (0.1) so it doesn't destabilize BFGS but can help if
    # the hidden function has slight curvature beyond cubic.
    sin_term = 0.1 * np.sin(x) + 0.05 * np.cos(2.0 * x)
    
    y = g1 + g2 + poly + sin_term
    
    return y
