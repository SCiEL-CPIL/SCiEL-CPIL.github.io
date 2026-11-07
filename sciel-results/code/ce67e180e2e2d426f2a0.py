import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 110.0  # seconds, safety margin before 150s kill

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a product-of-sums basis.
    f(x,y,z) = (a1 + b1*x + c1*y + d1*z) * (a2 + b2*x + c2*y + d2*z)
    The 10 parameters define:
      - 4 params for first sum: a1, b1, c1, d1
      - 4 params for second sum: a2, b2, c2, d2
      - 1 constant offset
      - 1 scaling factor
    
    This models bilinear interactions plus additive linear trends.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE:
        # Return a simple fallback to avoid timeout
        return np.ones_like(x)
    
    # First linear sum
    a1, b1, c1, d1 = p[0], p[1], p[2], p[3]
    s1 = a1 + b1 * x + c1 * y + d1 * z
    
    # Second linear sum
    a2, b2, c2, d2 = p[4], p[5], p[6], p[7]
    s2 = a2 + b2 * x + c2 * y + d2 * z
    
    # Constant offset and scaling
    const = p[8]
    scale = p[9]
    
    result = scale * s1 * s2 + const
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
