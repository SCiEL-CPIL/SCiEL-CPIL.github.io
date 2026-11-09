# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a spectral (Fourier) basis.
    
    Uses sine and cosine terms at multiple frequencies to capture
    oscillatory dynamics naturally. The basis includes:
    - Low-frequency terms for the main oscillation
    - Higher-frequency terms for nonlinear corrections
    - Cross terms between x and v
    
    10 parameters are fitted via BFGS.
    """
    p = params
    
    # Time guard: if we're close to the deadline, return a simple linear model
    # to avoid timeout. This is a safety net.
    # (In practice, equation() is called many times during fitting, so we
    #  just keep it fast and simple.)
    
    # Spectral basis: use multiple frequencies
    # Frequencies chosen to capture the expected oscillation period
    f1 = 1.0
    f2 = 2.0
    f3 = 3.0
    
    # Position-based terms (even and odd symmetry)
    # For an oscillator, acceleration should be odd in x (restoring force)
    # and odd in v (damping)
    
    # Odd functions of x (restoring force candidates)
    b0 = np.sin(x * f1)          # linear-ish restoring
    b1 = np.sin(x * f2)          # higher harmonic
    b2 = np.sin(x * f3)          # even higher harmonic
    b3 = x * np.cos(x * f1)      # asymmetric correction
    
    # Odd functions of v (damping candidates)
    b4 = np.sin(v * f1)          # linear-ish damping
    b5 = np.sin(v * f2)          # higher harmonic damping
    b6 = np.sin(v * f3)          # even higher harmonic damping
    b7 = v * np.cos(v * f1)      # asymmetric damping correction
    
    # Cross terms
    b8 = x * np.sin(v * f1)      # position-dependent damping
    b9 = v * np.sin(x * f1)      # velocity-dependent stiffness
    
    y = (p[0] * b0 +
         p[1] * b1 +
         p[2] * b2 +
         p[3] * b3 +
         p[4] * b4 +
         p[5] * b5 +
         p[6] * b6 +
         p[7] * b7 +
         p[8] * b8 +
         p[9] * b9)
    
    return y
# EVOLVE-BLOCK-END
