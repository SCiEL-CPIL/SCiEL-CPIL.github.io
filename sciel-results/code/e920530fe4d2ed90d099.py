"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a piecewise-linear basis built from absolute differences (V-shapes)
    to model yield points and work-hardening segments. Unlike ReLU-based
    kinks, these terms are active for all strains, ensuring gradients
    flow to breakpoint parameters even when initialized outside the
    data range (e.g., at 1.0 while max strain is ~0.96).
    
    params layout (10 constants):
      p0: base linear slope (Elastic modulus scale)
      p1, p2: weight and breakpoint for first kink
      p3, p4: weight and breakpoint for second kink
      p5, p6: weight and breakpoint for third kink
      p7, p8: weight and breakpoint for fourth kink
      p9: temperature linear coefficient
    """
    s = strain
    t = temp
    p = params
    
    p0 = p[0]
    w1 = p[1]
    b1 = p[2]
    w2 = p[3]
    b2 = p[4]
    w3 = p[5]
    b3 = p[6]
    w4 = p[7]
    b4 = p[8]
    p9 = p[9]
    
    # V-shaped kinks: abs(s - b)
    # These allow the model to fit both slope increases and decreases
    # and are always differentiable with respect to b.
    
    kink1 = w1 * np.abs(s - b1)
    kink2 = w2 * np.abs(s - b2)
    kink3 = w3 * np.abs(s - b3)
    kink4 = w4 * np.abs(s - b4)
    
    # Base linear term and temperature effect
    stress = p0 * s + kink1 + kink2 + kink3 + kink4 + p9 * t
    
    return stress
# EVOLVE-BLOCK-END
