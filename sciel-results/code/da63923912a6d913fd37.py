import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Quadratic Polynomial + 1 Adaptive Gaussian RBF
    Parameters:
    p[0]: w0 (constant)
    p[1]: w1 (x)
    p[2]: w2 (y)
    p[3]: w3 (x^2)
    p[4]: w4 (y^2)
    p[5]: w5 (xy)
    p[6]: w6 (RBF weight)
    p[7]: cx_raw (center x, transformed by tanh)
    p[8]: cy_raw (center y, transformed by tanh)
    p[9]: sigma_raw (width, transformed by exp)
    """
    # Extract parameters
    w0 = params[0]
    w1 = params[1]
    w2 = params[2]
    w3 = params[3]
    w4 = params[4]
    w5 = params[5]
    w6 = params[6]
    
    # Transform center and width to ensure valid ranges
    cx = np.tanh(params[7])
    cy = np.tanh(params[8])
    sigma = np.exp(params[9])
    
    # Prevent division by zero or overflow in exp if sigma is tiny
    # sigma is at least exp(-inf) -> 0, but params are finite.
    # If sigma is very small, exp(-dist^2 / (2*sigma^2)) can underflow to 0, which is fine.
    # If sigma is very large, it becomes 1, which is fine.
    
    # Calculate RBF term
    dx = x - cx
    dy = y - cy
    dist_sq = dx**2 + dy**2
    
    # Safe division
    # 2 * sigma^2
    denom = 2.0 * sigma * sigma
    if denom < 1e-12:
        rbf_term = np.zeros_like(x)
        # Only non-zero if dist is 0, but prob 0.
        rbf_term[np.isclose(dist_sq, 0.0)] = 1.0
    else:
        rbf_term = np.exp(-dist_sq / denom)
        
    pred = w0 + w1 * x + w2 * y + w3 * x**2 + w4 * y**2 + w5 * x * y + w6 * rbf_term
    
    return pred
