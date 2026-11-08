"""Rewritten program for symbolic regression of a nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with driving force, from position and velocity.
    Uses a physically-motivated, smooth basis with tanh saturation to improve
    optimizer conditioning and capture Duffing-like stiffness and nonlinear damping.
    """
    p = params
    
    # Scale inputs to roughly O(1) for better conditioning
    # x range: ~[-0.57, 0.53], v range: ~[-0.18, 0.28]
    # We'll use moderate scaling factors embedded in the basis
    xs = x * 2.0  # brings x to ~[-1.1, 1.0]
    vs = v * 5.0  # brings v to ~[-0.9, 1.4]
    
    # Restore force basis (odd, saturating)
    # tanh-based for smooth saturation
    f1 = np.tanh(xs)
    f2 = xs * np.tanh(xs * xs)  # cubic-like with saturation
    f3 = xs * xs * xs  # raw cubic (Duffing)
    
    # Damping force basis (odd in v, saturating)
    d1 = np.tanh(vs)
    d2 = vs * np.tanh(vs * vs)  # cubic-like damping with saturation
    d3 = vs * vs * vs  # raw cubic damping
    
    # Cross-coupling terms
    c1 = xs * vs  # linear coupling
    c2 = np.tanh(xs) * np.tanh(vs)  # saturated coupling
    
    # Asymmetric / even terms (smaller contribution expected)
    a1 = xs * xs  # quadratic in x
    a2 = vs * vs  # quadratic in v
    
    # Bias
    b = 1.0
    
    # Combine with 10 parameters
    # p[0]: linear-ish restoring (tanh)
    # p[1]: cubic-ish restoring (saturated)
    # p[2]: raw cubic restoring (Duffing)
    # p[3]: linear-ish damping (tanh)
    # p[4]: cubic-ish damping (saturated)
    # p[5]: raw cubic damping
    # p[6]: linear coupling
    # p[7]: saturated coupling
    # p[8]: quadratic in x (asymmetry)
    # p[9]: bias
    
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
