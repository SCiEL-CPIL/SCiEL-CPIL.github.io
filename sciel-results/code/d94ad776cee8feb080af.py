# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form NumPy expression for the acceleration of a nonlinear oscillator.
    
    Uses up to 10 fitted constants (params).
    The model uses a Duffing-like structure with:
    - Linear restoring and damping
    - Cubic nonlinear stiffness (Duffing term)
    - Quadratic damping
    - Mixed nonlinear terms
    
    Inputs are centered and scaled to improve numerical conditioning for BFGS.
    
    Model:
    a = p0 + p1*xs + p2*vs + p3*xs^3 + p4*vs^2 + p5*xs*vs + 
        p6*xs^2*vs + p7*vs^3 + p8*xs^4 + p9*vs*xs^2
    
    where xs = (x - x_center) / x_scale, vs = (v - v_center) / v_scale
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Center and scale inputs based on training ranges
    # x range: [-0.5728, 0.5275], center ~ -0.02265, half-range ~ 0.55015
    # v range: [-0.1754, 0.2821], center ~ 0.05335, half-range ~ 0.22875
    x_center = -0.02265
    x_scale = 0.55015
    v_center = 0.05335
    v_scale = 0.22875
    
    xs = (x - x_center) / x_scale
    vs = (v - v_center) / v_scale
    
    # Precompute powers
    xs2 = xs * xs
    xs3 = xs2 * xs
    xs4 = xs2 * xs2
    vs2 = vs * vs
    vs3 = vs2 * vs
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p
    
    # Construct the expression with Duffing-like structure
    # p0: constant offset
    # p1: linear restoring (stiffness)
    # p2: linear damping
    # p3: cubic nonlinear stiffness (Duffing)
    # p4: quadratic damping
    # p5: mixed linear term
    # p6: quadratic x, linear v
    # p7: cubic damping
    # p8: quartic x (softening/hardening)
    # p9: another mixed term for flexibility
    
    term = (p0 + 
            p1 * xs + 
            p2 * vs + 
            p3 * xs3 + 
            p4 * vs2 + 
            p5 * xs * vs + 
            p6 * xs2 * vs + 
            p7 * vs3 + 
            p8 * xs4 + 
            p9 * vs * xs2)
            
    return term
# EVOLVE-BLOCK-END
