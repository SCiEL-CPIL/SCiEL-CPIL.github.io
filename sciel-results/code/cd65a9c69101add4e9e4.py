# EVOLVE-BLOCK-START
import numpy as np


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    We use a mixture of two geometric distributions on integer support.
    One component decays fast (concentrating mass near 0), the other decays slow (heavy tail).
    This exploits the asymmetry: X1,X2,X3 should be small, X4 should be large.
    """
    n = 64
    support = np.arange(n, dtype=np.float64)
    
    # Mixture of two geometric distributions
    # Geometric prob: (1-r) * r^i
    r1 = 0.85  # Fast decay
    r2 = 0.98  # Slow decay
    p = 0.7    # Weight for fast decay component
    
    w1 = (1 - r1) * (r1 ** support)
    w2 = (1 - r2) * (r2 ** support)
    
    weights = p * w1 + (1 - p) * w2
    
    # Normalize
    weights = weights / np.sum(weights)
    
    return support, weights


# EVOLVE-BLOCK-END
