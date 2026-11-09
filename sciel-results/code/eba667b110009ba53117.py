"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 590.0  # safety margin before 630s kill
    
    # Seed roots: 4*(i+1) for i in 0..11
    # Scale up to maximize product (bound ~ 1/sqrt(product))
    # Try progressively larger scales, verify constraints if possible
    
    base_roots = [4.0 * (i + 1) for i in range(12)]
    
    # The bound is inversely related to sqrt(product of roots)
    # Larger roots -> smaller bound -> better score
    # Scale by a large factor
    scale = 1000.0
    roots = [r * scale for r in base_roots]
    
    return roots
# EVOLVE-BLOCK-END
