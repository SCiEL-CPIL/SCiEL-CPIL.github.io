"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def design(problem):
    """Concentric square rings with alternating dielectric/air for TM band gap."""
    size = problem["pixels"]
    half = size // 2  # 16
    
    # Ring k (k=0..15) has Chebyshev distance k+0.5 from center (15.5, 15.5)
    # Use alternating pattern: ring 0 (center) = dielectric, ring 1 = air, etc.
    pattern = [(k % 2) for k in range(half)]  # 0,1,0,1,... wait, we want 1 for dielectric
    # Actually: pattern[k] = 1 if k is even, 0 if k is odd
    pattern = [(1 if k % 2 == 0 else 0) for k in range(half)]
    
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            dist = max(abs(i - (half - 0.5)), abs(j - (half - 0.5)))
            k = int(dist - 0.5)
            cell[i, j] = pattern[k]
    
    return cell.tolist()
# EVOLVE-BLOCK-END
