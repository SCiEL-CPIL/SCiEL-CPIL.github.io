import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s kill

def priority(vector, n):
    # Check deadline guard
    if time.time() > _DEADLINE:
        # Return a neutral value; this should rarely trigger
        # since priority is called per-vector in a loop
        return 0.0
    
    # Count occurrences of 0, 1, 2
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Ideal count is n/3
    ideal = n / 3.0
    
    # Deviation from ideal balance (lower is better)
    dev = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    
    # Deterministic polynomial hash for tie-breaking
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val) % 1000000007
    
    # Normalize hash to [0, 1)
    tie_breaker = h / 1000000007.0
    
    # Primary: minimize deviation (negate for max priority)
    # Secondary: small hash component to break symmetries
    # Max dev for n=10 is ~66, so 0.01 is safe as epsilon
    return -dev + 0.01 * tie_breaker
