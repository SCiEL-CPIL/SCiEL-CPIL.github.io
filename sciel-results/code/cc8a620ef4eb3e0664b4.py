# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form NumPy expression for the acceleration of a nonlinear oscillator.
    
    Uses up to 10 fitted constants (params).
    Structural change: Switches from pure polynomial to a hybrid trigonometric/algebraic basis.
    Includes input scaling to improve numerical conditioning for BFGS.
    
    Model:
    a = p0 + p1*xs + p2*vs + p3*sin(xs) + p4*(cos(xs)-1) + 
        p5*sin(vs) + p6*(cos(vs)-1) + p7*xs*vs + p8*xs^2*vs + p9*xs*vs^2
    
    where xs = x / 0.6 and vs = v / 0.3
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Scale inputs to approximately [-1, 1] range for better conditioning
    xs = x / 0.6
    vs = v / 0.3
    
    # Precompute trigonometric and power terms
    sin_xs = np.sin(xs)
    cos_xs_m1 = np.cos(xs) - 1.0
    sin_vs = np.sin(vs)
    cos_vs_m1 = np.cos(vs) - 1.0
    
    xs2 = xs * xs
    vs2 = vs * vs
    
    # Linear and cross terms
    term0 = p[0]
    term1 = p[1] * xs
    term2 = p[2] * vs
    term3 = p[3] * sin_xs
    term4 = p[4] * cos_xs_m1
    term5 = p[5] * sin_vs
    term6 = p[6] * cos_vs_m1
    term7 = p[7] * xs * vs
    term8 = p[8] * xs2 * vs
    term9 = p[9] * xs * vs2
    
    return term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
# EVOLVE-BLOCK-END
