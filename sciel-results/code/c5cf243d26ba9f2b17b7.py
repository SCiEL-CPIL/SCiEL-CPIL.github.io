import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Van der Pol + Duffing nonlinear oscillator acceleration model.
    
    Structure: a = f(x, v) with:
      - Duffing restoring: linear + cubic in x
      - Van der Pol damping: linear + (x^2 - mu)*v, giving amplitude-dependent damping
      - Higher-order corrections: x^2*v, v^2, x*v, x^4, v^4, x^2*v^2
    
    Parameters (10):
      p0: linear restoring (omega^2)
      p1: cubic restoring (Duffing beta)
      p2: linear damping (gamma)
      p3: Van der Pol mu (amplitude-dependent damping)
      p4: x^2*v coupling
      p5: v^2 term
      p6: x*v coupling
      p7: x^4 restoring
      p8: v^3 damping
      p9: x^2*v^2 coupling
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers for clarity and efficiency
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    x2v = x2 * v
    x2v2 = x2 * v2
    
    # Duffing restoring force: linear + cubic + quartic
    restoring = p0 * x + p1 * x3 + p7 * x4
    
    # Van der Pol style damping: linear + amplitude-dependent
    # (x^2 - mu) * v gives energy injection at small amplitudes, dissipation at large
    damping = p2 * v + p3 * (x2 * v - v) + p4 * x2v + p8 * v3
    
    # Additional nonlinear couplings
    coupling = p5 * v2 + p6 * xv + p9 * x2v2
    
    result = restoring + damping + coupling
    
    return result
