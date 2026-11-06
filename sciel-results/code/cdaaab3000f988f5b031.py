import time

# EVOLVE-BLOCK-START

_DEADLINE = time.time() + 320
_priority_cache = {}

def _mix_hash(vector, n):
    """High-quality hash for tiebreaking."""
    h = 0x811c9dc5
    for x in vector:
        h ^= (x + 1)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h / 0xFFFFFFFF

def priority(vector, n):
    """
    Returns a priority score for the vector. Higher = should be considered earlier.
    
    Strategy: 
    Primary: -sum of squares of coordinates (vectors with smaller sum of squares get higher priority).
    Secondary: High-quality hash tiebreaker.
    """
    if time.time() > _DEADLINE:
        return 0.0
    
    key = (vector, n)
    if key in _priority_cache:
        return _priority_cache[key]
    
    # Sum of squares
    ss = sum(x * x for x in vector)
    
    # Hash tiebreaker
    h = _mix_hash(vector, n)
    
    # Combine: sum of squares dominates, hash breaks ties
    # Scale sum of squares to [0, 4n]
    score = -ss + 0.001 * h
    
    _priority_cache[key] = score
    return score

# EVOLVE-BLOCK-END
