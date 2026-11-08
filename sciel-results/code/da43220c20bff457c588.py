"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.special import roots_laguerre

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 uncertainty bound.
    
    Uses the zeros of the Laguerre polynomial L_12(x), which are the natural
    optimal root locations for the Cohn-Goncalves construction.
    """
    # Compute the 12 zeros of the Laguerre polynomial L_12(x)
    # roots_laguerre(n) returns the n roots of L_n(x)
    roots = roots_laguerre(12)
    
    # Ensure they are positive and sorted
    roots = sorted([float(r) for r in roots if r > 0])
    
    # Should have exactly 12 positive roots
    assert len(roots) == 12, f"Expected 12 roots, got {len(roots)}"
    
    return roots
# EVOLVE-BLOCK-END
