"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    n = p - 1  # 1008
    
    # Compute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 2
    best_len = 0
    
    # For each possible start position, find the maximum valid end
    for a in range(1, n + 1):
        # The interval must start at 'a'.
        # We extend 'b' as far as possible.
        # If [a, b] is invalid, [a, b+1] is also invalid.
        # So we stop at the first b that makes the interval invalid.
        
        # Check if 'a' itself is valid as a single element interval
        # Single element x is valid iff x*x % p != 1.
        # x*x % 1009 == 1 implies x == 1 or x == 1008.
        if a == 1 or a == 1008:
            continue
            
        # Initialize b to a
        b = a
        # We need to check if adding b to the interval [a, b-1] is valid.
        # For b=a, the interval is [a,a]. Valid if a not in {1, 1008}.
        
        while b <= n:
            # Check if adding 'b' creates a conflict
            # 1. Self-inverse check: b == 1 or b == 1008
            if b == 1 or b == 1008:
                break
            
            # 2. Pair check: inv[b] must not be in [a, b-1]
            # If inv[b] is in [a, b-1], then there exists x in [a, b-1] such that x*b == 1
            if a <= inv[b] < b:
                break
            
            # If we are here, [a, b] is a valid interval
            current_len = b - a + 1
            if current_len > best_len:
                best_len = current_len
                best_start = a
            
            b += 1
            
    return [best_start, best_len]
# EVOLVE-BLOCK-END
