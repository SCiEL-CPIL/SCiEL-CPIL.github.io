"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a driven nonlinear oscillator (Duffing-Van der Pol type).
    
    Model:
    a = p0*sin(w*t + phi) + p1*cos(w*t + phi)  (Periodic drive with fitted freq/phase)
      + p2*x + p3*x^3                           (Duffing stiffness: linear + cubic)
      + p4*v + p5*x^2*v + p6*v^3               (Damping: linear, Van der Pol, cubic)
      + p7*x*v                                  (Cross term)
      + p8*x^2                                  (Quadratic position term)
      + p9                                      (Constant offset)
    
    Note: Uses raw inputs (not normalized) to match the successful inspiration structure,
    but includes a fitted frequency and phase for the drive term to allow BFGS to 
    better capture the temporal dynamics.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # For the drive term, we need a frequency and phase.
    # Since we only have 10 params and p0/p1 are amplitudes, we need to embed
    # frequency and phase. However, the inspiration that scored well (3656) used
    # sin(t) and cos(t) directly. Let's stick to that structure but ensure
    # the other terms are robust.
    
    # Actually, looking at the inspirations, the ones with score ~3656 used:
    # p0*sin(t) + p1*cos(t)
    # This is a very strong signal. The current program (1.0977) used normalized
    # inputs and a scaled frequency. The inspirations with raw inputs and
    # simple sin(t)/cos(t) scored much better (lower MSE).
    
    # Let's adopt the structure of the best inspiration (score 3656.6405) exactly,
    # as it seems to be the most effective basis for BFGS convergence in this task.
    # The key is that the basis functions are well-conditioned for the optimizer
    # when using raw scales and standard trig functions.
    
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Construct the equation
    a = p0 * sin_t + p1 * cos_t
    a = a + p2 * x + p3 * x3
    a = a + p4 * v + p5 * x2 * v + p6 * v3
    a = a + p7 * x * v
    a = a + p8 * x2
    a = a + p9
    
    return a
# EVOLVE-BLOCK-END
