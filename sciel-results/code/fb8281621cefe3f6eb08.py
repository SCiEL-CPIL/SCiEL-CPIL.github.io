"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_DEADLINE = None

# Adaptive parameters
_HISTORY_SIZE = 200
_item_history = []
_typical_item_size = 1.0  # Initial guess
_last_update_time = time.time()

_INFEASIBLE = -1e9


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin based on a 'waste-avoidance' lookahead heuristic combined
    with consolidation bias.
    
    We penalize bins that leave a remaining capacity in a 'dead zone' where it
    is unlikely to fit future items, thereby reducing fragmentation. We also
    maintain a small bias toward already-used bins to encourage consolidation.
    """
    global _DEADLINE, _item_history, _typical_item_size, _last_update_time
    
    # Deadline guard
    if _DEADLINE is None:
        _DEADLINE = time.time() + 115.0  # 5s safety margin
    
    # Update typical item size periodically to avoid overhead on every call
    current_time = time.time()
    if current_time > _DEADLINE:
        # Fallback: return best-fit scores if we're running out of time
        bins = np.asarray(bins, dtype=float)
        remaining = bins - item
        feasible = remaining >= 0.0
        score = np.where(feasible, -remaining, _INFEASIBLE)
        return score
    
    # Update history and typical size
    _item_history.append(item)
    if len(_item_history) > _HISTORY_SIZE:
        _item_history.pop(0)
    
    # Update typical size every few calls to amortize cost
    if current_time - _last_update_time > 0.005:
        _typical_item_size = np.mean(_item_history) if _item_history else item
        _last_update_time = current_time
    
    bins = np.asarray(bins, dtype=float)
    remaining = bins - item
    feasible = remaining >= 0.0

    # Base score: Prefer tighter fit (smaller remaining)
    # We use -remaining so that 0 remaining is high score.
    score = -remaining.copy()
    
    # Lookahead Waste Penalty:
    # If the remaining space is in a "dead zone", it's effectively wasted.
    # Dead zone definition: 
    #   Lower bound: small fraction of typical item size
    #   Upper bound: larger fraction of typical item size
    # We estimate the next item size to be close to the typical size.
    
    # Use typical item size for more stable dead zone estimation
    # but clamp it to avoid extreme values
    typical = _typical_item_size
    if typical < 1e-6:
        typical = item
    
    # Dead zone: remaining space that is too small to be useful but too large to ignore
    # Scale based on typical item size
    dead_lower = 0.05 * typical
    dead_upper = 0.4 * typical
    
    # Create a mask for bins that fall into the dead zone
    in_dead_zone = (remaining > dead_lower) & (remaining < dead_upper) & feasible
    
    # Apply a strong penalty to these bins
    # The penalty should be enough to make them less attractive than a slightly looser fit
    # that leaves either very small waste (< dead_lower) or very large waste (> dead_upper)
    # which might fit another item.
    penalty = 3.0 * typical  # Strong penalty
    
    score = score - np.where(in_dead_zone, penalty, 0.0)
    
    # Consolidation bias: Still prefer bins that are already used to some extent
    # to reduce the number of active bins, but secondary to waste avoidance.
    # The waste avoidance already encourages filling bins to near capacity,
    # so this might be redundant or even counter-productive if it pushes into dead zones.
    # However, a small bias helps break ties in favor of "fuller" bins which are 
    # more likely to be closed soon.
    # Let's keep a very small consolidation bias.
    score = score + 0.02 * bins
    
    # Mask infeasible bins
    score = np.where(feasible, score, _INFEASIBLE)
    
    return score
# EVOLVE-BLOCK-END
