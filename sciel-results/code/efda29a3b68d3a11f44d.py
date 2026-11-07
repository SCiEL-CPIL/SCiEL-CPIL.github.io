import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid nonlinear oscillator model: Van der Pol / Duffing structure
    enriched with smooth nonlinear basis functions (tanh, sin) for
    bounded nonlinear corrections.
    
    Structure: a = f(x, v)
      - Duffing restoring: linear + cubic in x
      - Van der Pol damping: linear + (x^2 - 1)*v
      - Smooth nonlinear corrections via tanh and sin for bounded behavior
      - Velocity-dependent terms
    
    Parameters (10):
      p0: linear restoring (omega^2)
      p1: cubic restoring (Duffing beta)
      p2: linear damping (gamma)
      p3: Van der Pol mu (amplitude-dependent damping)
      p4: tanh(x)*v nonlinear damping correction
      p5: v^2 velocity-dependent stiffness
      p6: x*v parametric coupling
      p7: tanh(x) restoring correction
      p8: sin(v) damping correction
      p9: constant bias
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    
    # Smooth nonlinear basis functions
    tx = np.tanh(x)
    sv = np.sin(v)
    
    # Duffing restoring force: linear + cubic
    term_restore = p0 * x + p1 * x3 + p7 * tx
    
    # Van der Pol damping: linear + amplitude-dependent
    # (x^2 - 1)*v gives energy injection at small x, dissipation at large x
    term_vdP_damping = p2 * v + p3 * (x2 - 1.0) * v + p4 * tx * v + p8 * sv
    
    # Velocity-dependent stiffness and coupling
    term_vel_stiff = p5 * v2
    term_param_coup = p6 * x * v
    
    # Constant
    term_const = p9 * np.ones_like(x)
    
    result = term_restore + term_vdP_damping + term_vel_stiff + term_param_coup + term_const
    
    return result
