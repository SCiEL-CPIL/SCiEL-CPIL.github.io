"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration modeled using bounded non-linear activations (tanh)
    and a sinusoidal driving term.
    
    This approach differs from polynomial expansions by using hyperbolic tangent functions
    to model saturation and non-linearity in the restoring and damping forces, which is
    often more physically plausible for oscillators with bounded energy or amplitude.
    
    Parameters:
    p[0]: Gain for position-dependent restoring force
    p[1]: Scale for position-dependent restoring force
    p[2]: Gain for velocity-dependent damping force
    p[3]: Scale for velocity-dependent damping force
    p[4]: Amplitude of time-dependent driving force
    p[5]: Frequency of time-dependent driving force
    p[6]: Phase of time-dependent driving force
    p[7]: Gain for position-velocity cross-term
    p[8]: Scale for position-velocity cross-term
    p[9]: Constant offset
    """
    p = params
    
    # Nonlinear restoring force (saturation model)
    force_x = p[0] * np.tanh(p[1] * x)
    
    # Nonlinear damping force (saturation model)
    force_v = p[2] * np.tanh(p[3] * v)
    
    # Time-dependent driving force
    drive = p[4] * np.sin(p[5] * t + p[6])
    
    # Cross-coupling term (e.g., gyroscopic or parametric modulation)
    cross = p[7] * np.tanh(p[8] * x * v)
    
    # Offset
    offset = p[9]
    
    return force_x + force_v + drive + cross + offset
# EVOLVE-BLOCK-END
