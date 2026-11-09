import numpy as np
import time

_DEADLINE = time.time() + 140.0


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a normalized, bounded feature basis.
    
    Combines:
    - Normalized inputs (from inspiration 2/3) for BFGS conditioning
    - Bounded nonlinearities via tanh (from current program)
    - Complex-exponential phase-locked cross terms (structural change per directive)
    
    The complex representation z = x_n + i*v_n naturally couples position and
    velocity in a phase-locked manner. We extract real-valued basis functions
    from |z|^2, Re(z^2), Im(z^2), etc., which encode x*v coupling without
    the poorly-conditioned explicit x*v term.
    
    Features (10 params):
    p0: bias
    p1: linear restoring (x_n)
    p2: bounded nonlinear restoring (x_n * tanh(x_n))
    p3: linear damping (v_n)
    p4: bounded nonlinear damping (v_n * tanh(v_n))
    p5: sin(t) drive
    p6: cos(t) drive
    p7: phase-locked coupling Re(z^2) = x_n^2 - v_n^2
    p8: phase-locked coupling Im(z^2) = 2*x_n*v_n
    p9: quadratic velocity (v_n^2)
    """
    # Normalize inputs to unit scale for BFGS conditioning
    # t: [30, 50] -> center 40, scale 10
    # x: [-0.2056, 0.1749] -> center -0.015, scale 0.2
    # v: [-0.3391, 0.3624] -> center 0.01, scale 0.35
    x_n = (x + 0.015) / 0.2
    v_n = (v - 0.01) / 0.35
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Bounded nonlinear features via tanh
    x_tanh = np.tanh(x_n)
    v_tanh = np.tanh(v_n)
    
    # Bounded restoring and damping
    x_nonlin = x_n * x_tanh
    v_nonlin = v_n * v_tanh
    
    # Periodic driving
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Complex-exponential derived cross terms:
    # z = x_n + i*v_n
    # Re(z^2) = x_n^2 - v_n^2  (phase-locked, encodes relative phase)
    # Im(z^2) = 2*x_n*v_n      (phase-locked coupling)
    # These are more structured than raw x*v and help BFGS converge
    re_z2 = x_n * x_n - v_n * v_n
    im_z2 = 2.0 * x_n * v_n
    
    # Quadratic velocity for energy-dependent effects
    v2 = v_n * v_n
    
    # Linear combination of basis functions
    result = (
        p0
        + p1 * x_n
        + p2 * x_nonlin
        + p3 * v_n
        + p4 * v_nonlin
        + p5 * sin_t
        + p6 * cos_t
        + p7 * re_z2
        + p8 * im_z2
        + p9 * v2
    )
    
    return result
