import math
import time

# EVOLVE-BLOCK-START
def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    
    # Calculate LCM
    L = 1
    for d in divisors:
        L = L * d // math.gcd(L, d)
        
    best_range = float('inf')
    best_S = 0
    
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    # Iterate over all possible starting points in one period
    for S in range(L):
        if time.time() > deadline:
            break
            
        m = []
        for d in divisors:
            # Smallest multiple of d >= S
            q = (S + d - 1) // d
            m.append(q * d)
            
        # Sort to handle distinctness
        m_sorted = sorted(m)
        
        # Resolve conflicts greedily to find minimal range for this S
        used = set()
        adjusted = []
        for val in m_sorted:
            v = val
            while v in used:
                v += L
            used.add(v)
            adjusted.append(v)
            
        current_range = adjusted[-1] - adjusted[0]
        
        if current_range < best_range:
            best_range = current_range
            best_S = S

    # Compute final values for best_S
    base_multiples = []
    for d in divisors:
        q = (best_S + d - 1) // d
        base_multiples.append(q * d)
        
    # Assign values to specific divisors
    # Sort indices by their base multiple
    indexed = sorted([(m, i) for i, m in enumerate(base_multiples)])
    
    assigned = [0] * len(divisors)
    used_vals = set()
    
    for m, idx in indexed:
        v = m
        while v in used_vals:
            v += L
        used_vals.add(v)
        assigned[idx] = v
        
    return assigned
# EVOLVE-BLOCK-END
