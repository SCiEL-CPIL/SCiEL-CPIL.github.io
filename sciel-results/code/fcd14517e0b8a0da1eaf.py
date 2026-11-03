"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 50

# EVOLVE-BLOCK-START
def construct():
    # The ratio is scale-dependent or constrained.
    # Assuming L1=1 or L-inf=1 constraint, sparse vectors maximize the ratio.
    # A single spike gives the highest possible ratio (1.0) under L1=1 normalization.
    # Under L-inf=1, a single spike of height 1 gives ratio 1.0.
    # The uniform vector gives ~0.02.
    # Thus, a single spike is a massive improvement.
    
    # Try a single spike at index 0
    h = [0.0] * N
    h[0] = 1.0
    
    return h
# EVOLVE-BLOCK-END
