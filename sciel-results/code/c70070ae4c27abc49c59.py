"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0
    
    p = 1009
    n = p - 1  # 1008
    
    # Compute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        
    # For each x, determine the interval [low, high] that x and inv[x] span.
    # If an interval [a, b] contains both x and inv[x], it is invalid.
    # This happens if a <= low and b >= high.
    # For a fixed start a, the maximum valid b is min(high(x) for all x with low(x) >= a) - 1.
    
    # Let block[a] = min(high(x)) for all x such that low(x) == a.
    # Then H[a] = min(block[a], H[a+1]).
    # The max length starting at a is H[a] - a.
    
    INF = 10**9
    block = [INF] * (n + 2)
    
    for x in range(1, n + 1):
        y = inv[x]
        low = min(x, y)
        high = max(x, y)
        if high < block[low]:
            block[low] = high
            
    # Compute H[a] from right to left
    # H[a] is the minimum high of any pair (x, inv[x]) with low >= a.
    H = [INF] * (n + 2)
    H[n+1] = INF
    for a in range(n, 0, -1):
        H[a] = min(block[a], H[a+1])
        
    best_start = 2
    best_len = 0
    
    for a in range(1, n + 1):
        if time.time() > deadline:
            break
            
        # If H[a] is INF, it means no pair with low >= a exists? 
        # But (1,1) has low=1, (1008,1008) has low=1008.
        # For a=1008, H[1008] = 1008. Length = 1008-1008=0.
        # For a=1007, H[1007] = min(block[1007], 1008).
        
        # The interval [a, b] is valid if b < H[a].
        # So max b is H[a] - 1.
        # Length is (H[a] - 1) - a + 1 = H[a] - a.
        
        length = H[a] - a
        
        # We must also ensure the interval is within [1, 1008].
        # Since H[a] <= 1008 (because of the pair 1008,1008 which has low=1008, high=1008, 
        # for any a <= 1008, H[a] <= 1008), b_max <= 1007.
        # So the upper bound is naturally handled.
        # The lower bound is a >= 1.
        
        # Check if a is a valid start (not self-inverse, etc.)
        # If a is self-inverse, low=a, high=a, so block[a] <= a, so H[a] <= a, length <= 0.
        # So self-inverse starts naturally yield length 0 or negative.
        
        if length > best_len:
            best_len = length
            best_start = a
            
    return [best_start, best_len]
# EVOLVE-BLOCK-END
