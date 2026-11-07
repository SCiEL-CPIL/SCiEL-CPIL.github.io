import numpy as np

# EVOLVE-BLOCK-START

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model using Gaussian RBF basis in (x, v) space
    plus harmonic time terms.
    
    a = A1 * exp(-((x_s - cx1)^2 + (v_s - cv1)^2) / (2*sigma^2)) + 
        A2 * exp(-((x_s - cx2)^2 + (v_s - cv2)^2) / (2*sigma^2)) + 
        p7*(t-40) + p8*cos(t) + p9*sin(t)
        
    Inputs are scaled to O(1) for spatial terms.
    """
    # Scale inputs to O(1) for better conditioning
    # x range [-0.2056, 0.1749] -> scale by 0.2
    # v range [-0.3391, 0.3624] -> scale by 0.35
    
    X = x / 0.2
    V = v / 0.35
    
    # Extract params
    A1 = params[0]
    cx1 = params[1]
    cv1 = params[2]
    A2 = params[3]
    cx2 = params[4]
    cv2 = params[5]
    sigma = params[6]
    # Avoid division by zero or negative sigma
    sigma = np.abs(sigma) + 1e-6
    
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Gaussian terms
    d1_sq = (X - cx1)**2 + (V - cv1)**2
    g1 = np.exp(-d1_sq / (2.0 * sigma**2))
    
    d2_sq = (X - cx2)**2 + (V - cv2)**2
    g2 = np.exp(-d2_sq / (2.0 * sigma**2))
    
    # Time terms
    # Center t to avoid large values in linear term
    T_lin = t - 40.0
    time_term = p7 * T_lin + p8 * np.cos(t) + p9 * np.sin(t)
    
    pred = A1 * g1 + A2 * g2 + time_term
    
    return pred

# EVOLVE-BLOCK-END
