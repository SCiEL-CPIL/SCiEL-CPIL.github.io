import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model using Van der Pol-inspired basis.
    
    The Van der Pol oscillator has the form:
        a = x - mu*(1 - x^2)*v
    
    This captures amplitude-dependent damping: energy is injected at small
    amplitudes (x^2 < 1) and dissipated at large amplitudes (x^2 > 1).
    
    We extend this with higher-order nonlinear terms to capture more complex
    dynamics while maintaining physical interpretability:
        - p0: constant offset
        - p1: linear restoring force (spring constant)
        - p2: linear damping
        - p3: cubic nonlinear restoring force (hard/soft spring)
        - p4: velocity-squared damping (quadratic drag)
        - p5: position-velocity coupling (Van der Pol core)
        - p6: quintic nonlinear restoring force
        - p7: cubic velocity damping
        - p8: x^2*v term (amplitude-dependent damping correction)
        - p9: quartic position term
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    v2 = v * v
    v3 = v2 * v
    
    xv = x * v
    x2v = x2 * v
    x3v = x3 * v
    
    # Basis functions
    t1 = np.ones_like(x)       # constant
    t2 = x                     # linear position
    t3 = v                     # linear velocity (damping)
    t4 = x3                    # cubic nonlinear spring
    t5 = v2                    # quadratic drag
    t6 = xv                    # position-velocity coupling
    t7 = x5                    # quintic nonlinear spring
    t8 = v3                    # cubic velocity damping
    t9 = x2v                   # amplitude-dependent damping (Van der Pol)
    t10 = x4                   # quartic position term
    
    return (p0 * t1 + p1 * t2 + p2 * t3 + p3 * t4 + 
            p4 * t5 + p5 * t6 + p6 * t7 + p7 * t8 + 
            p8 * t9 + p9 * t10)
