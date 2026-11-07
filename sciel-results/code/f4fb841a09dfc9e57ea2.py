# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE = 145.0  # seconds, 5s safety margin before 150s kill

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational-Affine Model with Centering.
    
    Model: f(x, y, z) = scale * (N(x, y, z) / D(x, y, z)) + offset
    
    Where N and D are affine functions of centered inputs.
    
    Parameters:
      p[0..3]  : Numerator coefficients (const, cx, cy, cz)
      p[4..7]  : Denominator coefficients (const, cx, cy, cz)
      p[8]     : Scale factor
      p[9]     : Offset
    """
    # Deadline check
    if time.time() > _DEADLINE:
        x_arr = np.asarray(x, dtype=np.float64)
        p_arr = np.asarray(params, dtype=np.float64)
        return np.ones_like(x_arr) * float(p_arr.flat[0])
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.shape != (10,):
        p = np.resize(p, 10)
    
    # Center inputs to improve numerical stability and conditioning
    # Range [0.5, 2.0] -> Center at 1.25 -> Range [-0.75, 0.75]
    x_c = x - 1.25
    y_c = y - 1.25
    z_c = z - 1.25
    
    # Numerator: p[0] + p[1]*x_c + p[2]*y_c + p[3]*z_c
    num = p[0] + p[1] * x_c + p[2] * y_c + p[3] * z_c
    
    # Denominator: p[4] + p[5]*x_c + p[6]*y_c + p[7]*z_c
    den = p[4] + p[5] * x_c + p[6] * y_c + p[7] * z_c
    
    # Safe division to prevent NaN/Inf
    # If denominator is very close to zero, the function value becomes huge.
    # We use a small epsilon to avoid exact division by zero, but allow large values.
    # The optimizer will penalize large values via MSE.
    eps = 1e-8
    den_safe = np.where(np.abs(den) < eps, np.sign(den) * eps + (den == 0) * eps, den)
    
    ratio = num / den_safe
    
    # Final output: scale * ratio + offset
    result = p[8] * ratio + p[9]
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
# EVOLVE-BLOCK-END
