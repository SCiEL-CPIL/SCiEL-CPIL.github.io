"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp
import time

# EVOLVE-BLOCK-START
def construct():
    """Return six coefficients for the growth-index approximation."""
    # Optimized coefficients for CPL dark-energy cosmologies
    c0 = 0.55
    c1 = 0.05
    c2 = -0.02
    c3 = 0.03
    c4 = 0.0
    c5 = -0.01
    return [c0, c1, c2, c3, c4, c5]
# EVOLVE-BLOCK-END
