import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model for a driven nonlinear oscillator.
    Basis functions:
    1. Constant offset
    2. Linear stiffness (x)
    3. Cubic stiffness (x^3) - Duffing
    4. Linear damping (v)
    5. Nonlinear damping (x^2 * v) - Van der Pol
    6. Quadratic drag (v * |v|)
    7. Sine driving (sin(t))
    8. Cosine driving (cos(t))
    9. Coupling (x * v)
    10. Quadratic stiffness (x^2)
    """
    p = params
    
    # State terms
    term_x = x
    term_x3 = x**3
    term_x2 = x**2
    term_v = v
    term_vdp = x**2 * v
    term_vdrag = v * np.abs(v)
    term_xv = x * v
    
    # Time/Drive terms
    term_sin_t = np.sin(t)
    term_cos_t = np.cos(t)
    
    # Constant
    term_const = np.ones_like(x)
    
    return (p[0] * term_const + 
            p[1] * term_x + 
            p[2] * term_x3 + 
            p[3] * term_v + 
            p[4] * term_vdp + 
            p[5] * term_vdrag + 
            p[6] * term_sin_t + 
            p[7] * term_cos_t + 
            p[8] * term_xv + 
            p[9] * term_x2)
