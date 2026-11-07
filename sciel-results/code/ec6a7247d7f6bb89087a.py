import numpy as np
from scipy.optimize import minimize

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a radial basis function (RBF) representation.
    
    The 10 parameters are organized as:
    - params[0:5]: x-coordinates of 5 Gaussian centers in the (x, v) plane
    - params[5:10]: v-coordinates of 5 Gaussian centers in the (x, v) plane
    
    Wait, that's only 10 params for centers. Let me reconsider.
    
    Actually, with 10 params total, I'll use:
    - 5 RBF terms with fixed centers and widths, and 5 fitted coefficients
    OR
    - Use params to define centers and widths in a structured way
    
    Let me use: 5 centers (cx_i, cy_i) where cx_i = params[2*i], cy_i = params[2*i+1]
    And fixed widths. This uses all 10 params for center locations.
    """
    # Scale inputs to roughly [-1, 1] for better conditioning
    X = x / 0.6
    V = v / 0.3
    
    # 5 Gaussian RBF centers defined by params
    # Center i: (params[2*i], params[2*i+1])
    # Fixed widths for stability
    width = 0.5
    
    n_rbf = 5
    result = np.zeros_like(x)
    
    for i in range(n_rbf):
        cx = params[2 * i]
        cy = params[2 * i + 1]
        
        # Gaussian RBF: exp(-((X-cx)^2 + (V-cy)^2) / (2*width^2))
        dx = X - cx
        dy = V - cy
        dist2 = dx * dx + dy * dy
        gaussian = np.exp(-dist2 / (2.0 * width * width))
        
        # Fixed coefficient for each RBF term (alternating signs for flexibility)
        coeff = 1.0 if i % 2 == 0 else -1.0
        result += coeff * gaussian
    
    # Scale output to match target range approximately
    # Target range is [-0.1047, 0.1106], so scale by ~0.1
    result = result * 0.1
    
    return result
