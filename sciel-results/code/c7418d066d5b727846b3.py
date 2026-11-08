"""Rewritten program for symbolic regression of a nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with driving force, from position and velocity.
    Uses a physically-motivated, smooth basis with tanh saturation for good conditioning.
    Designed so all basis terms are O(1) over the training range and have low mutual correlation.
    """
    p = params
    
    # Scale inputs to O(1) for good conditioning
    # x range: ~[-0.57, 0.53] -> scale by 2 to get ~[-1.1, 1.0]
    # v range: ~[-0.18, 0.28] -> scale by 4 to get ~[-0.7, 1.1]
    xs = x * 2.0
    vs = v * 4.0
    
    # Restoring force basis (odd in x, smooth, well-separated)
    # p[0]: linear restoring (tanh gives ~linear for small args, saturates)
    f1 = np.tanh(xs)
    # p[1]: higher-order restoring (x*tanh(x^2) is ~x^3 for small x, saturates)
    f2 = xs * np.tanh(xs * xs)
    # p[2]: pure cubic Duffing term (bounded by input range, no saturation needed at this scale)
    f3 = xs * xs * xs
    
    # Damping force basis (odd in v, smooth)
    # p[3]: linear damping
    d1 = np.tanh(vs)
    # p[4]: cubic-like damping
    d2 = vs * np.tanh(vs * vs)
    # p[5]: pure cubic damping
    d3 = vs * vs * vs
    
    # Cross terms (smaller expected contribution)
    # p[6]: linear coupling
    c1 = xs * vs
    # p[7]: saturated coupling
    c2 = np.tanh(xs) * np.tanh(vs)
    
    # Even/asymmetric terms (smaller expected contribution)
    # p[8]: quadratic in x (asymmetry)
    a1 = xs * xs
    # p[9]: bias
    b = 1.0
    
    y = (p[0] * f1 +
         p[1] * f2 +
         p[2] * f3 +
         p[3] * d1 +
         p[4] * d2 +
         p[5] * d3 +
         p[6] * c1 +
         p[7] * c2 +
         p[8] * a1 +
         p[9] * b)
    
    return y
# EVOLVE-BLOCK-END
