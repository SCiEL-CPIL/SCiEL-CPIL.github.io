import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Parametric Composite Basis.
    
    Strategy:
    - Unlike the previous Chebyshev approach which used a fixed linear basis, 
      this approach uses a non-linear parametric model.
    - The 10 parameters define:
      1. 3 Gaussian RBFs: (center_i, width_i, amplitude_i) for i=1..3
      2. A 3rd-degree polynomial trend: a0 + a1*x + a2*x^2 + a3*x^3
    - This allows the model to capture both global trends (polynomial) and 
      local non-linear features (Gaussians) simultaneously.
    - The non-linearity in the parameters (centers and widths) creates a 
      multi-modal optimization landscape, but provides a much richer 
      hypothesis space for functions that are not well-approximated by 
      low-degree polynomials alone.
    - We use a softplus activation for widths to ensure positivity and 
      differentiability.
    """
    # Deadline guard
    start_time = time.time()
    
    # Ensure x is 1D
    x = np.asarray(x).flatten()
    n = len(x)
    
    # Extract parameters
    # p[0:3] -> centers c1, c2, c3
    # p[3:6] -> raw widths w1, w2, w3 (will be transformed)
    # p[6:9] -> amplitudes A1, A2, A3
    # p[9]   -> polynomial coefficient a3 (x^3)
    # We need 4 polynomial coeffs but only have 1 param left.
    # Let's adjust: 
    # Use 2 Gaussians (4 params: c1, w1, c2, w2) and 1 Amplitude param for both? 
    # No, better to use the 10 params more effectively.
    
    # Revised Parameter Allocation (10 params):
    # 2 Gaussians:
    #   p[0], p[1]: center1, raw_width1
    #   p[2], p[3]: center2, raw_width2
    #   p[4]:       amplitude1
    #   p[5]:       amplitude2
    # 4th Degree Polynomial (5 coeffs? No, we have 4 params left p[6:9] + p[9])
    #   p[6]: a0
    #   p[7]: a1
    #   p[8]: a2
    #   p[9]: a3
    # Total: 6 + 4 = 10 params.
    
    c1 = params[0]
    w1_raw = params[1]
    c2 = params[2]
    w2_raw = params[3]
    A1 = params[4]
    A2 = params[5]
    a0 = params[6]
    a1 = params[7]
    a2 = params[8]
    a3 = params[9]
    
    # Softplus for widths to ensure positive and smooth
    # softplus(x) = log(1 + exp(x))
    # To avoid overflow in exp, use stable version or clip
    # Since params start at 1, softplus(1) ~ 1.31. 
    # If params go large, exp overflows. 
    # Alternative: 1 + exp(w) is always > 1. Or 1/sigmoid(w) ?
    # Let's use a safe softplus: 
    # np.logaddexp(0, w) is stable.
    w1 = np.logaddexp(0.0, w1_raw)
    w2 = np.logaddexp(0.0, w2_raw)
    
    # Compute Gaussians
    # G(x) = A * exp(-((x-c)^2) / (2*w^2))
    # To prevent underflow/overflow issues in exp, we can clip the exponent.
    # However, for x in [-2,2] and reasonable c,w, it should be fine.
    
    diff1 = x - c1
    diff2 = x - c2
    
    # Clip exponents to avoid extreme values if w is very small
    # If w is very small, diff^2/w^2 can be huge.
    # exp(-huge) -> 0. This is fine.
    # But if w is 0, division by zero. Softplus ensures w >= log(1+exp(-inf)) = 0? 
    # logaddexp(0, -1000) is approx 0. 
    # If w is extremely close to 0, we might get instability.
    # Let's enforce a minimum width, e.g., 1e-3.
    w1 = np.maximum(w1, 1e-3)
    w2 = np.maximum(w2, 1e-3)
    
    exp1 = -0.5 * (diff1 / w1) ** 2
    exp2 = -0.5 * (diff2 / w2) ** 2
    
    # Clip to avoid underflow warnings, though numpy handles it
    exp1 = np.clip(exp1, -700, 700)
    exp2 = np.clip(exp2, -700, 700)
    
    g1 = A1 * np.exp(exp1)
    g2 = A2 * np.exp(exp2)
    
    # Polynomial part
    poly = a0 + a1 * x + a2 * x**2 + a3 * x**3
    
    # Combine
    result = poly + g1 + g2
    
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    # Time check (though this is very fast)
    if time.time() - start_time > 110.0:
        return result
        
    return result
