import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    rem = bins - item
    valid = rem >= 0
    scores = np.full_like(rem, -np.inf, dtype=float)
    scores[valid] = rem[valid]
    return scores
