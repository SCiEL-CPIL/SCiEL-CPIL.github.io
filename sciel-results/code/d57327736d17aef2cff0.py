"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0  # seconds before the 150s limit
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

# Polarization parameters
_CAPACITY = 100.0
# We want to encourage bins to be either nearly full or nearly empty
# Penalty peaks at mid-fill (50% of capacity used)
_POLARIZATION_WEIGHT = 0.3
_GAP_WEIGHT = 0.7


def _polarization_score(remaining_after: np.ndarray) -> np.ndarray:
    """
    Score based on how 'polarized' the bin would be after placement.
    Higher score = more polarized (closer to 0 or capacity).
    
    After placing item, used = bins - remaining_after
    We want used close to 0 or close to _CAPACITY.
    """
    used = _CAPACITY - remaining_after
    # Distance from ideal polarization points (0 and _CAPACITY)
    # Minimize distance to nearest ideal point
    dist_to_0 = np.abs(used)
    dist_to_cap = np.abs(_CAPACITY - used)
    min_dist = np.minimum(dist_to_0, dist_to_cap)
    # Normalize to [0, 1] where 1 is most polarized
    polarization = 1.0 - min_dist / (_CAPACITY / 2.0)
    return polarization


def _gap_score(remaining_after: np.ndarray) -> np.ndarray:
    """
    Best-fit style: prefer smaller remaining capacity (tighter fit).
    """
    # Smaller remaining = better, so negate
    return -remaining_after


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Polarization-aware hybrid scoring.
    Combines best-fit gap minimization with polarization to avoid
    creating inflexible half-full bins.
    
    Score = _GAP_WEIGHT * normalized_gap + _POLARIZATION_WEIGHT * polarization
    """
    # Check if we're close to the deadline
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        # Emergency: return a simple fallback (pure best-fit)
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores
    
    remaining_after = bins - item
    feasible = remaining_after >= 0
    
    if not np.any(feasible):
        return np.full(len(bins), -np.inf)
    
    scores = np.full(len(bins), -np.inf)
    
    # For feasible bins, compute hybrid score
    rem_feasible = remaining_after[feasible]
    
    # Gap component: prefer tighter fit (smaller remaining)
    # Normalize gap to [0, 1] where 1 is tightest fit
    max_rem = np.max(rem_feasible) if len(rem_feasible) > 0 else 1.0
    min_rem = np.min(rem_feasible) if len(rem_feasible) > 0 else 0.0
    if max_rem > min_rem:
        gap_norm = 1.0 - (rem_feasible - min_rem) / (max_rem - min_rem)
    else:
        gap_norm = np.ones_like(rem_feasible)
    
    # Polarization component
    pol = _polarization_score(rem_feasible)
    
    # Hybrid score
    hybrid = _GAP_WEIGHT * gap_norm + _POLARIZATION_WEIGHT * pol
    
    scores[feasible] = hybrid
    
    return scores
# EVOLVE-BLOCK-END
