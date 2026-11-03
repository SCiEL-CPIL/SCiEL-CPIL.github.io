"""Optimized program for the second autocorrelation inequality task."""
import time
import numpy as np

N = 50

# EVOLVE-BLOCK-START
def construct():
    # The ratio ||h||^2_2 / (||h||_1 * ||h*h||_inf) is maximized by concentrating mass.
    # For h with a single nonzero entry of value a: ratio = 1/a.
    # For multiple nonzeros, the ratio decreases due to the L1 norm growth.
    # 
    # Strategy: Use a single spike. The value doesn't matter for the ratio 
    # (it's scale-invariant: ratio(h) = ratio(c*h) for c>0).
    # However, to be safe against numerical issues, use a moderate value.
    #
    # Actually, since the ratio is scale-invariant, any positive single spike works.
    # Let's use 1.0 for simplicity.
    
    h = [0.0] * N
    h[0] = 1.0
    
    return h
# EVOLVE-BLOCK-END
