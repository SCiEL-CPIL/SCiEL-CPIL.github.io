import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores

    remaining_after = bins - item
    feasible = remaining_after >= 0

    if not np.any(feasible):
        return np.full(len(bins), -np.inf)

    scores = np.full(len(bins), -np.inf)
    
    rem = remaining_after[feasible]
    
    # Polarization Strategy:
    # Avoid bins that end up with "middle" remaining space (around 50).
    # Prefer bins that are either very full (rem close to 0) or very empty (rem close to 100).
    # Score is high for extremes, low for the middle.
    # We add a small bias towards empty bins (higher rem) in case of ties or near-ties,
    # as empty bins offer more future flexibility.
    
    # Distance from the "bad" middle (50)
    polarization_score = np.abs(rem - 50.0)
    
    # Weight the polarization heavily
    score = 10.0 * polarization_score + rem
    
    scores[feasible] = score
    
    return scores
