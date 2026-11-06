"""Cap set priority function for greedy insertion."""
import time
import hashlib

_start_time = time.time()
_DEADLINE = 300  # seconds, with safety margin

# Cache for computed priorities to avoid recomputation
_priority_cache = {}

def priority(vector, n):
    """
    Assign priority to a vector in {0,1,2}^n for greedy cap set construction.
    Higher priority = added earlier.
    
    Strategy: Balance symbol counts to reduce early constraint saturation,
    with secondary criteria for diversity and structure.
    """
    global _start_time, _DEADLINE
    
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return 0.0
    
    # Check cache
    vkey = vector
    if vkey in _priority_cache:
        return _priority_cache[vkey]
    
    # Count symbols
    count0 = 0
    count1 = 0
    count2 = 0
    for v in vector:
        if v == 0:
            count0 += 1
        elif v == 1:
            count1 += 1
        else:
            count2 += 1
    
    # Balance score: how close to n/3 each count is
    # Lower deviation = better balance = higher priority
    target = n / 3.0
    deviation = abs(count0 - target) + abs(count1 - target) + abs(count2 - target)
    
    # Primary: minimize deviation (higher priority for more balanced)
    # Convert to priority: negative deviation so lower deviation = higher priority
    balance_priority = -deviation
    
    # Secondary: prefer vectors with more "spread" in positions
    # Use a simple hash for tiebreaking to ensure diversity
    # Convert vector to string for hashing
    vstr = ''.join(str(v) for v in vector)
    h = int(hashlib.md5(vstr.encode()).hexdigest(), 16)
    hash_priority = (h % 1000000) / 1000000.0  # [0, 1)
    
    # Tertiary: prefer vectors with more 1s in middle positions (structural)
    # Simple structural score: sum of position * value
    struct_score = sum((i + 1) * v for i, v in enumerate(vector))
    struct_priority = struct_score / (n * 2.0)  # normalize to [0, 1]
    
    # Combine: balance is dominant, hash for tiebreaking, structure as minor
    # Scale balance to dominate
    total_priority = balance_priority * 1000.0 + hash_priority * 10.0 + struct_priority * 1.0
    
    _priority_cache[vkey] = total_priority
    return total_priority
