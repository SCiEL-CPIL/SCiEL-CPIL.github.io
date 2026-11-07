# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form NumPy expression for the acceleration of a nonlinear oscillator.
    
    Uses up to 10 fitted constants (params).
    The model includes linear terms, quadratic non-linearities in x and v,
    and mixed terms to capture potential non-linear restoring forces and
    non-linear damping.
    
    To improve numerical stability for the BFGS optimizer, the inputs x and v
    are scaled to be approximately in the range [-1, 1] before applying the
    polynomial expansion. This prevents higher-order terms from becoming
    very small (ill-conditioned) relative to lower-order terms.
    
    Model:
    a = p0 + p1*xs + p2*vs + p3*xs^2 + p4*vs^2 + p5*xs*vs + p6*xs^3 + p7*vs^3 + p8*xs^2*vs + p9*vs^2*xs
    
    where xs = x / 0.6 and vs = v / 0.3
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    # Ensure shapes are compatible
    p = np.asarray(params, dtype=np.float64)
    
    # Scale inputs to O(1) range to improve conditioning
    # Ranges: x ~ [-0.57, 0.53], v ~ [-0.18, 0.28]
    # Scaling factors chosen to normalize to approx [-1, 1]
    x_scale = 0.6
    v_scale = 0.3
    
    xs = x / x_scale
    vs = v / v_scale
    
    # Extract parameters
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]
    p4 = p[4]
    p5 = p[5]
    p6 = p[6]
    p7 = p[7]
    p8 = p[8]
    p9 = p[9]
    
    # Compute terms
    xs2 = xs * xs
    vs2 = vs * vs
    xs3 = xs2 * xs
    vs3 = vs2 * vs
    
    term = (p0 + 
            p1 * xs + 
            p2 * vs + 
            p3 * xs2 + 
            p4 * vs2 + 
            p5 * xs * vs + 
            p6 * xs3 + 
            p7 * vs3 + 
            p8 * xs2 * vs + 
            p9 * vs2 * xs)
            
    return term
# EVOLVE-BLOCK-END
