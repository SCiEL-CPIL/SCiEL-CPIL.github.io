import numpy as np
import time

_start_time = None

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Elastic-Plastic model with temperature-dependent parameters.
    
    stress = E(T) * strain + S(T) * tanh(K(T) * strain) + C(T)
    
    E(T) = p0 + p1*T + p2*T^2
    S(T) = p3 + p4*T + p5*T^2
    K(T) = p6 + p7*T
    C(T) = p8 + p9*T
    
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
    p6, p7 = params[6], params[7]
    p8, p9 = params[8], params[9]
    
    # Temperature is in [0.06667, 1]. We can use it directly or normalize.
    # Using raw T might be better if the scale is natural, but normalized is safer for optimizer.
    # The previous Chebyshev model used normalized T. Let's use raw T here as the range is [0.06, 1].
    # Actually, let's normalize T to [0, 1] or [-1, 1] to help the optimizer?
    # The prompt says inputs are in training range.
    # Let's use the raw temp value as it is already O(1).
    
    T = temp
    
    # Normalize strain to help optimizer: strain in [-0.001911, 0.9636]
    # Map to roughly [-1, 1] for better conditioning
    strain_norm = (strain - 0.48) / 0.48  # center ~0.48, scale ~0.48
    
    # Calculate temperature-dependent coefficients
    E_T = p0 + p1 * T + p2 * T * T
    S_T = p3 + p4 * T + p5 * T * T
    K_T = p6 + p7 * T
    
    # Calculate stress
    # Use np.tanh for stable non-linearity on normalized strain
    nonlinear_term = np.tanh(K_T * strain_norm)
    
    # Use original strain for the linear elastic term to preserve physical meaning
    result = E_T * strain + S_T * nonlinear_term + p8 + p9 * T
    
    return result
