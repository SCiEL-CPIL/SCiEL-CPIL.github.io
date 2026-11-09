import numpy as np
import sympy as sp
import time

def construct():
    """
    Returns 12 double-root locations z_1..z_12.
    Uses the zeros of the Bessel function J_{1/2}(x), which are (k*pi)^2.
    """
    # The zeros of J_{1/2}(x) are at x = (k*pi)^2 for k=1, 2, ...
    # We need 12 distinct positive finite locations.
    k = np.arange(1, 13)
    z_floats = (k * np.pi) ** 2
    
    # Ensure they are sorted and distinct
    z_floats = np.sort(z_floats)
    
    return list(z_floats)
