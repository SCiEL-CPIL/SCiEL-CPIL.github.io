import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: 2 Gaussians (with bounded centers) + Cubic Polynomial.
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
    
    # Squash centers to [-2, 2] to keep them within the domain of interest
    # tanh(1) ~ 0.76, tanh(-1) ~ -0.76. 
    # To cover [-2, 2] better, we can scale: 2 * tanh(c_raw)
    c1 = 2.0 * np.tanh(c1_raw)
    c2 = 2.0 * np.tanh(c2_raw)
    
    # Widths
    # Clip log_w to prevent extreme widths
    # w = exp(log_w). If log_w is 5, w=148 (very wide). If -5, w=0.006 (very narrow).
    log_w1 = np.clip(log_w1, -5, 5)
    log_w2 = np.clip(log_w2, -5, 5)
    w1 = np.exp(log_w1)
    w2 = np.exp(log_w2)
    
    # Calculate Gaussians
    g1_arg = -0.5 * ((x - c1) / w1) ** 2
    g2_arg = -0.5 * ((x - c2) / w2) ** 2
    
    # Clip to avoid underflow (exp(-50) is ~1e-22, safe)
    g1_arg = np.clip(g1_arg, -50, 50)
    g2_arg = np.clip(g2_arg, -50, 50)
    
    g1 = a1 * np.exp(g1_arg)
    g2 = a2 * np.exp(g2_arg)
    
    # Calculate Polynomial
    poly = b0 + b1 * x + b2 * x**2 + b3 * x**3
    
    # Sum
    y = g1 + g2 + poly
    
    return y
