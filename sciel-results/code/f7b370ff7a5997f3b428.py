import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Van der Pol / Trigonometric nonlinear oscillator model.
    
    Combines the amplitude-dependent damping of the Van der Pol oscillator
    with trigonometric phase-dependent modulation.
    
    Parameters (10):
      p0: linear restoring (omega^2)
      p1: cubic restoring (Duffing beta)
      p2: linear damping (base gamma)
      p3: amplitude-dependent damping (Van der Pol: p3*x^2*v)
      p4: trigonometric coupling (sin(x)*v)
      p5: cubic damping (v^3)
      p6: velocity-dependent stiffness (v^2*x)
      p7: cross-damping (x^2*v^2)
      p8: higher-order restoring (x^5)
      p9: bias / offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    x5 = x2 * x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Restoring forces (potential derivatives)
    restore = p0 * x + p1 * x3 + p8 * x5
    
    # Damping: Van der Pol structure => (p2 + p3*x2) * v + p5*v3
    # Plus trigonometric coupling p4 * sin(x) * v
    damping = (p2 + p3 * x2) * v + p5 * v3 + p4 * np.sin(x) * v
    
    # Cross/coupling terms
    coupling = p6 * v2 * x + p7 * x2 * v2
    
    # Constant
    offset = p9 * np.ones_like(x)
    
    result = restore + damping + coupling + offset
    
    return result
