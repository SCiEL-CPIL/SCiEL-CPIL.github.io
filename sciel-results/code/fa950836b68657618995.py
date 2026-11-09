"""Physically-motivated multiplicative nonlinear oscillator: parameters represent
stiffness, damping, and nonlinear coupling with distinct physical roles, giving BFGS
a well-conditioned landscape where each parameter has a clear gradient direction."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 135.0
_GLOBAL = {}


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multiplicative nonlinear oscillator model.
    
    Parameters (10 total) with physical interpretation:
    p0: baseline offset
    p1: linear stiffness (x term)
    p2: cubic stiffness (x^3 term) - hardening/softening spring
    p3: linear damping (v term)
    p4: velocity nonlinear damping (v^3 term)
    p5: position-velocity coupling (x*v term) - gyroscopic/Coriolis-like
    p6: time-dependent forcing amplitude (sin component)
    p7: time-dependent forcing frequency modifier
    p8: position-time interaction (t*x) - drift or parametric excitation
    p9: velocity-time interaction (t*v) - friction variation
    
    The multiplicative structure arises from:
    - Nonlinear spring: (p1*x + p2*x^3) creates a potential well
    - Nonlinear damping: (p3*v + p4*v^3) creates velocity-dependent drag
    - Cross-coupling: p5*x*v creates energy exchange between position and velocity
    - Time-varying terms: p6*sin(t), p7*cos(t) for periodic forcing
    - Parametric terms: p8*t*x, p9*t*v for time-dependent system properties
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Core nonlinear spring force (potential derivative)
    spring_force = p1 * x + p2 * x * x * x
    
    # Core nonlinear damping force (velocity-dependent drag)
    damping_force = p3 * v + p4 * v * v * v
    
    # Cross-coupling term (position-velocity interaction)
    coupling = p5 * x * v
    
    # Time-periodic forcing (sinusoidal drive)
    forcing = p6 * np.sin(t) + p7 * np.cos(t)
    
    # Parametric time-varying terms
    parametric = p8 * t * x + p9 * t * v
    
    # Combine: the target is presumably the acceleration or a related quantity
    # For an oscillator, target might be: a = f(x, v, t)
    # We model the full nonlinear dynamics
    pred = p0 + spring_force + damping_force + coupling + forcing + parametric
    
    return pred
# EVOLVE-BLOCK-END
