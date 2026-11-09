import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a multiplicative structure with a polynomial surface for temp/pH.
    
    Model: y = A * Monod(s) * Inhibit(b) * P(temp, pH) + B
    P(temp, pH) is a 2nd-order polynomial in normalized temperature and pH.
    
    Parameters (10):
    p0: Global Amplitude
    p1: Monod Ks
    p2: Density Inhibition Constant
    p3: Polynomial offset
    p4: Linear Temp coeff
    p5: Linear pH coeff
    p6: Quadratic Temp coeff
    p7: Quadratic pH coeff
    p8: Interaction coeff
    p9: Additive Bias
    """
    t0 = time.time()
    
    # Extract parameters
    A = params[0]
    K_s = np.abs(params[1]) + 1e-6
    K_b = np.abs(params[2]) + 1e-6
    c0 = params[3]
    c1 = params[4]
    c2 = params[5]
    c3 = params[6]
    c4 = params[7]
    c5 = params[8]
    B = params[9]
    
    # Normalize inputs to help optimization
    # temp range approx [1, 40], center 20, span 20
    t_norm = (temp - 20.0) / 20.0
    # pH range approx [1, 12], center 6.5, span 5.5
    h_norm = (pH - 6.5) / 5.5
    
    # Monod kinetics for substrate
    # s range [1, 40]
    monod = s / (K_s + s + 1e-12)
    
    # Density inhibition
    # b range [1, 5000]
    # Using a simple saturation/inhibition term: 1 / (1 + b/K_b)
    dens_term = 1.0 / (1.0 + b / K_b)
    
    # Polynomial surface
    # P = c0 + c1*t + c2*h + c3*t^2 + c4*h^2 + c5*t*h
    poly = c0 + c1 * t_norm + c2 * h_norm + c3 * t_norm**2 + c4 * h_norm**2 + c5 * t_norm * h_norm
    
    # Combine
    result = A * monod * dens_term * poly + B
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if time.time() - t0 > 140:
        return result
        
    return result
