"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    t_start = time.time()
    # Keep a safety margin to ensure we return within the 60s/90s limit
    # The prompt says 60s evaluation, but budget is 90s.
    # We will use 85s as a hard stop to be safe.
    DEADLINE = t_start + 85.0
    
    N = 4096
    S = []
    sums = set()
    
    for x in range(1, N + 1):
        if time.time() > DEADLINE:
            break
        
        # Generate all new triple sums that include x
        # The triples are (x,x,x), (x,x,y), (x,y,y), (x,y,z) with y<z
        new_sums = []
        
        # (x, x, x)
        new_sums.append(3 * x)
        
        # (x, x, y) and (x, y, y)
        for y in S:
            new_sums.append(2 * x + y)
            new_sums.append(x + 2 * y)
            
        # (x, y, z) for y < z in S
        # S is sorted because we iterate x upwards and append
        for i in range(len(S)):
            y = S[i]
            for j in range(i + 1, len(S)):
                z = S[j]
                new_sums.append(x + y + z)
                
        # Check if new sums are unique among themselves
        if len(new_sums) != len(set(new_sums)):
            continue
            
        # Check if new sums are disjoint from existing sums
        new_set = set(new_sums)
        if new_set.isdisjoint(sums):
            S.append(x)
            sums.update(new_set)
            
    return S
# EVOLVE-BLOCK-END
