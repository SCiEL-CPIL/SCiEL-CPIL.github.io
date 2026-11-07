"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multi-scale RBF ensemble strategy.
    
    The 10 parameters are grouped into 3 independent RBF 'centers':
    - Center 1: params[0] (center), params[1] (width), params[2] (weight)
    - Center 2: params[3] (center), params[4] (width), params[5] (weight)
    - Center 3: params[6] (center), params[7] (width), params[8] (weight)
    - params[9]: Global bias/offset
    
    This structure allows the optimizer to fit local non-linear features
    (peaks, valleys, oscillations) by adjusting the position and shape of
    Gaussian bumps, which is more flexible than a global polynomial basis
    and often converges better for localized features in symbolic regression.
    """
    # Unpack parameters
    c1, w1, a1 = params[0], params[1], params[2]
    c2, w2, a2 = params[3], params[4], params[5]
    c3, w3, a3 = params[6], params[7], params[8]
    bias = params[9]
    
    # Ensure widths are positive to avoid division by zero or negative variance
    # We use abs() to allow the optimizer to explore both sides of zero,
    # but the physical width must be positive.
    w1 = np.abs(w1) + 1e-6
    w2 = np.abs(w2) + 1e-6
    w3 = np.abs(w3) + 1e-6
    
    # Calculate RBF contributions
    # Using standard Gaussian form: weight * exp(-((x - center)^2) / (2 * width^2))
    rbf1 = a1 * np.exp(-((x - c1) ** 2) / (2.0 * w1 ** 2))
    rbf2 = a2 * np.exp(-((x - c2) ** 2) / (2.0 * w2 ** 2))
    rbf3 = a3 * np.exp(-((x - c3) ** 2) / (2.0 * w3 ** 2))
    
    # Combine
    y = rbf1 + rbf2 + rbf3 + bias
    
    return y
# EVOLVE-BLOCK-END
