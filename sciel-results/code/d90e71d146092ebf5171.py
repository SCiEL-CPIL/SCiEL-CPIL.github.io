"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Power-law best-fit with exponent 1.5: softer than quadratic but still
    # prefers tighter fits. This may reduce fragmentation for Weibull items.
    remaining = bins - item
    # Clip to avoid negative values (shouldn't happen if capacity is sufficient)
    remaining = np.maximum(remaining, 0.0)
    return -(remaining ** 1.5)
# EVOLVE-BLOCK-END
