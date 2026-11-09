"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    # Use the 12 smallest positive zeros of J_{1/2}(x) = sin(sqrt(x))/sqrt(x)
    # These are at x = (k*pi)^2 for k = 1, 2, ..., 12
    # This exploits asymptotic orthogonality of Laguerre polynomials to Bessel measure
    
    n = 12
    z = [(k * math.pi) ** 2 for k in range(1, n + 1)]
    
    # Ensure they are floats, positive, finite, and pairwise distinct
    z = [float(zi) for zi in z]
    z = sorted(z)
    
    return z
# EVOLVE-BLOCK-END
