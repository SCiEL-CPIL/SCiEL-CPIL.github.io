"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_START_TIME = None

def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    return _time.time() - _START_TIME

def _vector_hash(v, n):
    """Deterministic hash of vector for tie-breaking."""
    h = 0
    for i in range(n):
        h = (h * 31 + v[i]) & 0xFFFFFFFF
    return h

def priority(vector, n):
    """
    Priority based on structural cap-set heuristics:
    - Fewer 1s -> higher priority (less likely to be AP midpoint)
    - More balanced 0/2 distribution -> higher priority
    - Deterministic hash for tie-breaking
    """
    if _get_elapsed() > 280:
        return 0.0  # Safety: return neutral if over time
    
    # Count 1s
    n_ones = sum(1 for x in vector if x == 1)
    
    # Count 0s and 2s
    n_zeros = sum(1 for x in vector if x == 0)
    n_twos = sum(1 for x in vector if x == 2)
    
    # Balance score: prefer balanced 0/2
    balance = abs(n_zeros - n_twos)
    
    # Primary: fewer 1s is better (higher priority)
    # Secondary: more balanced 0/2 is better
    # Tertiary: hash for deterministic diversity
    h = _vector_hash(vector, n)
    
    # Scale: n_ones in [0, n], balance in [0, n], h in [0, 2^32)
    # We want higher priority = larger return value
    priority_val = (n - n_ones) * 1.0 - balance * 0.1 + (h % 10000) / 10000.0 * 0.01
    
    return priority_val
# EVOLVE-BLOCK-END
