import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    2-component Gaussian Mixture Model with linear trend.
    
    Parameters:
    p0, p1: mean_x1, mean_y1
    p2, p3: mean_x2, mean_y2
    p4, p5: log_sigma1, log_sigma2 (sigma = exp(log_sigma))
    p6, p7: amplitude1, amplitude2
    p8, p9: linear_coeff_x, linear_coeff_y
    
    Model:
    f(x, y) = b1*x + b2*y + 
              A1 * exp(-((x-mx1)^2 + (y-my1)^2) / (2*sigma1^2)) + 
              A2 * exp(-((x-mx2)^2 + (y-my2)^2) / (2*sigma2^2))
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64)
    
    # Unpack parameters
    mx1, my1, mx2, my2, log_s1, log_s2, A1, A2, b1, b2 = params
    
    # Compute sigmas safely
    # Clip log_sigma to prevent overflow in exp or underflow in division
    # exp(20) is ~ 4.8e8, exp(-20) is ~ 2e-9
    # If sigma is too small, exp(-d^2/sigma^2) becomes 0 quickly (step function-like)
    # If sigma is too large, it becomes 1 (constant)
    log_s1 = np.clip(log_s1, -10, 10)
    log_s2 = np.clip(log_s2, -10, 10)
    
    s1 = np.exp(log_s1)
    s2 = np.exp(log_s2)
    
    # Precompute squared distances
    dx1 = x - mx1
    dy1 = y - my1
    dist1_sq = dx1 * dx1 + dy1 * dy1
    
    dx2 = x - mx2
    dy2 = y - my2
    dist2_sq = dx2 * dx2 + dy2 * dy2
    
    # Compute Gaussian terms
    # Using exp(-0.5 * d^2 / s^2)
    term1 = A1 * np.exp(-0.5 * dist1_sq / (s1 * s1))
    term2 = A2 * np.exp(-0.5 * dist2_sq / (s2 * s2))
    
    # Linear terms
    lin_term = b1 * x + b2 * y
    
    pred = lin_term + term1 + term2
    
    # Safety check for non-finite values (should not happen with clipped sigmas)
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
