import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for the acceleration of a nonlinear oscillator.
    
    Uses a hybrid model:
    1. Physical basis: Linear restoring force, linear damping, and a cubic 
       nonlinear stiffness term (Duffing-like). This captures the essential 
       physics of a nonlinear oscillator with damping.
    2. Residual correction: A small set of Gaussian RBF terms centered at 
       representative points in the training domain to capture any remaining 
       structure not explained by the physical basis.
    
    Parameters:
    x: Position (np.ndarray)
    v: Velocity (np.ndarray)
    params: Fitted constants (np.ndarray, length 10)
    
    Returns:
    Acceleration (np.ndarray)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Physical basis terms
    # p0 * x: Linear restoring force
    # p1 * v: Linear damping
    # p2 * x^3: Nonlinear stiffness (Duffing term)
    x2 = x * x
    x3 = x2 * x
    
    base = p0 * x + p1 * v + p2 * x3
    
    # RBF residual terms
    # Centers chosen to cover the training domain:
    # x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
    # We use 3 Gaussian terms with fixed centers and bandwidths.
    # The parameters p3-p9 control the weights and potentially widths.
    # To keep it simple and well-conditioned, we fix the centers and widths,
    # and let BFGS fit the 7 remaining weights (p3..p9).
    
    # Center 1: Origin
    c1_x, c1_v = 0.0, 0.0
    # Center 2: Positive x, small positive v
    c2_x, c2_v = 0.3, 0.1
    # Center 3: Negative x, small negative v
    c3_x, c3_v = -0.3, -0.1
    
    # Fixed bandwidth (inverse of sigma squared)
    # Training ranges: dx ~ 1.1, dv ~ 0.45
    # A reasonable bandwidth might be around 10-20 for each dimension
    # Let's use a common bandwidth for simplicity, or separate ones.
    # Let's try separate bandwidths for better conditioning.
    # sigma_x ~ 0.3, sigma_v ~ 0.15
    # exp(-((x-cx)^2/sigma_x^2 + (v-cv)^2/sigma_v^2))
    
    # Precompute distances
    d1 = (x - c1_x)**2 / (0.3**2) + (v - c1_v)**2 / (0.15**2)
    d2 = (x - c2_x)**2 / (0.3**2) + (v - c2_v)**2 / (0.15**2)
    d3 = (x - c3_x)**2 / (0.3**2) + (v - c3_v)**2 / (0.15**2)
    
    # Gaussian kernels
    g1 = np.exp(-d1)
    g2 = np.exp(-d2)
    g3 = np.exp(-d3)
    
    # We have 7 params left (p3..p9). 
    # Let's use 3 for the weights of the Gaussians, and 4 more for 
    # additional local terms or just leave them as zero? 
    # Actually, let's use all 7 to be safe, but maybe 3 is enough.
    # If we only use 3, the other 4 params will be fitted to 0 or unused.
    # To avoid unused params causing issues, let's use them in a meaningful way.
    # How about 2 more Gaussian terms?
    
    # Center 4: Positive x, negative v
    c4_x, c4_v = 0.3, -0.1
    d4 = (x - c4_x)**2 / (0.3**2) + (v - c4_v)**2 / (0.15**2)
    g4 = np.exp(-d4)
    
    # Center 5: Negative x, positive v
    c5_x, c5_v = -0.3, 0.1
    d5 = (x - c5_x)**2 / (0.3**2) + (v - c5_v)**2 / (0.15**2)
    g5 = np.exp(-d5)
    
    # That's 5 Gaussians. We have 7 params.
    # Let's add 2 more centers or use the extra params for scaling the Gaussians?
    # Or maybe just use 5 Gaussians and 2 extra terms like x*v and x^2*v?
    
    # Let's stick to 5 Gaussians and use the remaining 2 params for 
    # interaction terms that might be missing from the base.
    # p8 * x * v
    # p9 * x**2 * v
    
    residual = p3 * g1 + p4 * g2 + p5 * g3 + p6 * g4 + p7 * g5 + p8 * (x * v) + p9 * (x2 * v)
    
    return base + residual
