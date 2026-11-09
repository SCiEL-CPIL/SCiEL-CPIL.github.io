# EVOLVE-BLOCK-START
import numpy as np


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    This implementation uses a 3-point distribution on {0, 1, 2} with optimized weights
    that yield a probability of approximately 0.2794, which is higher than the 
    uniform distribution on {0, ..., 31} (approx 0.2708).
    
    Returns:
        support: np.ndarray of shape (3,), non-negative support points.
        weights: np.ndarray of shape (3,), non-negative weights.
    """
    support = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    weights = np.array([0.5, 0.2, 0.3], dtype=np.float64)
    return support, weights


# EVOLVE-BLOCK-END
