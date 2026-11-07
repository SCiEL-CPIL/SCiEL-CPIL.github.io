"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with driving force, from position and velocity."""
    # Rescale inputs to improve conditioning
    # x range: [-0.5728, 0.5275], center ~ -0.02265, half-range ~ 0.55
    # v range: [-0.1754, 0.2821], center ~ 0.05335, half-range ~ 0.22875
    
    # Use fixed scaling based on training ranges to improve BFGS convergence
    # x_scaled = (x - center_x) / half_range_x
    # v_scaled = (v - center_v) / half_range_v
    
    # But we can't compute center/half_range dynamically in equation() without knowing the data
    # So we use approximate fixed scalings based on the given ranges
    # x ~ [-0.57, 0.53], so scale by ~0.55, shift by ~-0.02
    # v ~ [-0.18, 0.28], so scale by ~0.23, shift by ~0.05
    
    # Actually, let's just use simple scaling to unit magnitude
    # x / 0.55, v / 0.23
    
    xs = x / 0.55
    vs = v / 0.23
    
    # Nonlinear basis for oscillator dynamics
    # Common terms: x, x^3, v, v^3, x^2*v, x*v^2, x*v, bias
    # Also consider sin(x), sin(v) for smooth nonlinearities
    
    x2 = xs * xs
    v2 = vs * vs
    x3 = x2 * xs
    v3 = v2 * vs
    x2v = x2 * vs
    xv2 = xs * v2
    xv = xs * vs
    
    # Use trigonometric terms which are natural for oscillators
    sx = np.sin(xs)
    sv = np.sin(vs)
    
    # Combine terms - use all 10 params
    # params[0]: linear x
    # params[1]: cubic x  
    # params[2]: linear v
    # params[3]: cubic v
    # params[4]: x^2 * v
    # params[5]: x * v^2
    # params[6]: x * v
    # params[7]: sin(x)
    # params[8]: sin(v)
    # params[9]: bias
    
    return (params[0] * xs +
            params[1] * x3 +
            params[2] * vs +
            params[3] * v3 +
            params[4] * x2v +
            params[5] * xv2 +
            params[6] * xv +
            params[7] * sx +
            params[8] * sv +
            params[9])
# EVOLVE-BLOCK-END
