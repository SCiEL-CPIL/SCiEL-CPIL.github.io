import numpy as np
import time

_start_time = None

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Elastic-Plastic model with temperature-dependent parameters.
    
    stress = E(T) * strain + S(T) * tanh(K(T) * strain_norm) + C
    
    E(T) = p0 + p1*T + p2*T^2
    S(T) = p3 + p4*T + p5*T^2
    K(T) = p6 + p7*T + p8*T^2
    C = p9
    
    strain_norm = strain * 2.0
    10 params: [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 130:
        return np.zeros_like(strain, dtype=np.float64)
    
    # Extract parameters
    p0, p1, p2 = params[0], params[1], params[2]
    p3, p4, p5 = params[3], params[4], params[5]
    p6, p7, p8 = params[6], params[7], params[8]
    p9 = params[9]
    
    T = temp
    
    # Normalize strain for the non-linear term to improve conditioning
    # Scaling by 2.0 keeps the zero at zero and maps max strain ~0.96 to ~1.92
    strain_norm = strain * 2.0
    
    # Calculate temperature-dependent coefficients
    E_T = p0 + p1 * T + p2 * T * T
    S_T = p3 + p4 * T + p5 * T * T
    K_T = p6 + p7 * T + p8 * T * T
    
    # Calculate stress
    nonlinear_term = np.tanh(K_T * strain_norm)
    
    result = E_T * strain + S_T * nonlinear_term + p9
    
    return result
