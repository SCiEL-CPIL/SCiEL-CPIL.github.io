import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Precompute LCM of 1..24
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 25))
    
    # V[x] = L // x
    V = {x: L // x for x in range(1, 25)}
    
    S = []
    current_sums = {0}
    
    # Greedy: iterate from 24 down to 1
    for x in range(24, 0, -1):
        if time.time() > deadline:
            break
            
        # Check if V[x] is in current_sums
        # current_sums contains all subsums of {V[y] for y in S}
        # Note: S only contains elements > x because we iterate downwards
        # So the condition "1/x is not a subsum of reciprocals of OTHER selected elements"
        # is equivalent to V[x] not in subsums of S.
        # Wait, is it?
        # The condition is: 1/x != sum_{y in T} 1/y for T subset of S \ {x}.
        # Since we are checking before adding x, S does not contain x.
        # So S \ {x} is just S.
        # And all y in S are > x.
        # So yes, we just check if V[x] is a subsum of S.
        
        if V[x] not in current_sums:
            S.append(x)
            # Update current_sums
            # new_sums = current_sums U {s + V[x] for s in current_sums}
            # To avoid modifying while iterating, create a list of new sums
            new_sums = set()
            for s in current_sums:
                new_sums.add(s + V[x])
            current_sums.update(new_sums)
            
            # Prune if set gets too large to prevent memory/time explosion?
            # If len(current_sums) > 5_000_000, we might be in trouble.
            # But 2^23 is 8.4M.
            # If we exceed this, we can stop updating sums and just use a heuristic?
            # Or break.
            if len(current_sums) > 5_000_000:
                # If sums set is huge, further additions are unlikely to be valid?
                # Or we can just stop the greedy and return S.
                break

    # Local Search / Improvement
    # Try to add skipped elements by removing others?
    # For now, return S.
    
    if not S:
        return [24]
        
    return sorted(S)

# EVOLVE-BLOCK-END
