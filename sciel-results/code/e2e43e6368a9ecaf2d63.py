# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs (x, y) drawn from [-1, 1]^2.
    
    Strategy: Multi-scale RBF network with 3 RBF units + polynomial basis.
    - 3 RBF units, each with center (cx, cy), width (sx, sy), and amplitude
      That's 3 * 4 = 12 params... but we only have 10.
    
    Alternative: 2 RBF units + polynomial
    - RBF1: cx1, cy1, sx1, sy1, amp1 (5 params)
    - RBF2: cx2, cy2, sx2, sy2, amp2 (5 params)
    Total: 10 params, no polynomial.
    
    Or: 1 RBF + higher-order polynomial
    - RBF: cx, cy, sx, sy, amp (5 params)
    - Poly: 1, x, y, x^2, y^2, xy, x^3, y^3, x^2y, xy^2 (10 terms)
    But that's 15 params.
    
    Let's do: 2 RBF units with shared width + polynomial
    - RBF1: cx1, cy1, amp1 (3 params)
    - RBF2: cx2, cy2, amp2 (3 params)
    - Shared width: sx, sy (2 params)
    - Polynomial: 1, x, y (3 params)
    Total: 3+3+2+3 = 11... still too many.
    
    Let me reconsider. With 10 params:
    Option A: 2 RBF units, each with (cx, cy, sx, sy, amp) = 10 params, no polynomial
    Option B: 1 RBF (cx, cy, sx, sy, amp) = 5 params + 5-term polynomial (1, x, y, x^2, y^2)
    Option C: 3 RBF units with fixed centers but learnable amplitudes and shared width
    
    Let's go with Option A: 2 RBF units, each fully parameterized.
    This gives more local flexibility than the current single RBF.
    
    RBF1: exp(-((x-cx1)^2/sx1^2 + (y-cy1)^2/sy1^2)) * amp1
    RBF2: exp(-((x-cx2)^2/sx2^2 + (y-cy2)^2/sy2^2)) * amp2
    
    params[0:5] = cx1, cy1, sx1, sy1, amp1
    params[5:10] = cx2, cy2, sx2, sy2, amp2
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # RBF 1
    cx1 = p[0]
    cy1 = p[1]
    sx1 = np.abs(p[2]) + 0.1
    sy1 = np.abs(p[3]) + 0.1
    amp1 = p[4]
    
    dx1 = x - cx1
    dy1 = y - cy1
    rbf1_arg = -(dx1**2 / (sx1**2) + dy1**2 / (sy1**2))
    rbf1 = amp1 * np.exp(rbf1_arg)
    
    # RBF 2
    cx2 = p[5]
    cy2 = p[6]
    sx2 = np.abs(p[7]) + 0.1
    sy2 = np.abs(p[8]) + 0.1
    amp2 = p[9]
    
    dx2 = x - cx2
    dy2 = y - cy2
    rbf2_arg = -(dx2**2 / (sx2**2) + dy2**2 / (sy2**2))
    rbf2 = amp2 * np.exp(rbf2_arg)
    
    return rbf1 + rbf2
# EVOLVE-BLOCK-END
