"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a piecewise-linear "sawtooth" / zigzag basis built from abs and min
    to model yield points and work-hardening segments.
    
    params layout (10 constants):
      p0: base linear slope (Elastic modulus scale)
      p1: temperature linear coefficient
      p2: intercept
      p3, p4: first kink breakpoint and slope change
      p5, p6: second kink breakpoint and slope change
      p7, p8: third kink breakpoint and slope change
      p9: temperature interaction with strain (optional)
    """
    s = strain
    t = temp
    p = params
    
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]  # breakpoint 1
    p4 = p[4]  # slope change 1
    p5 = p[5]  # breakpoint 2
    p6 = p[6]  # slope change 2
    p7 = p[7]  # breakpoint 3
    p8 = p[8]  # slope change 3
    p9 = p[9]  # temp-strain interaction
    
    # Piecewise linear segments using abs:
    # max(0, x) = (x + abs(x)) / 2
    # This creates kinks at the breakpoints
    
    # Segment 1: active when s > p3
    seg1 = (s - p3 + np.abs(s - p3)) / 2.0
    # Segment 2: active when s > p5
    seg2 = (s - p5 + np.abs(s - p5)) / 2.0
    # Segment 3: active when s > p7
    seg3 = (s - p7 + np.abs(s - p7)) / 2.0
    
    # Base linear term
    base = p0 * s + p1 * t + p2
    
    # Add kinked contributions
    kink1 = p4 * seg1
    kink2 = p6 * seg2
    kink3 = p8 * seg3
    
    # Temperature-strain interaction
    interaction = p9 * s * t
    
    stress = base + kink1 + kink2 + kink3 + interaction
    
    return stress
# EVOLVE-BLOCK-END
