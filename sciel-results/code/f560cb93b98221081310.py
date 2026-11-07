import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Rational function form: (a1*x + a2*x^3 + a3*v + a4*x^2*v + a5*x*v) / (1 + b1*x^2 + b2*v^2 + b3*x^2*v^2)
    # This is a structurally different model family that can capture nonlinear dynamics
    # with saturating behavior, better suited for oscillator systems.
    
    # Normalize inputs
    x_scale = 0.21
    v_scale = 0.37
    
    xn = x / x_scale
    vn = v / v_scale
    
    # Pre-compute powers
    xn2 = xn * xn
    xn3 = xn2 * xn
    vn2 = vn * vn
    
    # Numerator terms (5 params)
    num = (
        params[0] * xn +          # linear restoring
        params[1] * xn3 +         # cubic restoring  
        params[2] * vn +          # linear damping
        params[3] * xn2 * vn +    # van der Pol coupling
        params[4] * xn * vn       # cross damping
    )
    
    # Denominator terms (5 params) - must be positive for stability
    den = (
        1.0 +
        params[5] * xn2 +         # position saturation
        params[6] * vn2 +         # velocity saturation
        params[7] * xn2 * vn2 +   # joint saturation
        params[8] * xn3 * vn +    # nonlinear cross term
        params[9] * xn * vn2      # velocity-dependent restoring
    )
    
    # Ensure denominator is well-conditioned (avoid division by near-zero)
    den_safe = np.where(np.abs(den) < 1e-8, np.sign(den) * 1e-8 + (1 - np.abs(np.sign(den))) * 1e-8, den)
    
    result = num / den_safe
    
    return result
