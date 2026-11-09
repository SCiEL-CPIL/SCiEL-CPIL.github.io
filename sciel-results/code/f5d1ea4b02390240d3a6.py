import random
import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # LCM of 1..24
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 25))
    
    # Precompute integer reciprocals: val[x] = L // x
    val = {x: L // x for x in range(1, 25)}
    
    best_set = list(range(13, 25)) # Fallback
    best_size = len(best_set)
    
    def greedy(order):
        nonlocal best_set, best_size
        
        S = []
        sums = {0}
        
        for x in order:
            vx = val[x]
            
            # Check if vx is in sums (violation for x)
            if vx in sums:
                continue
            
            # Calculate new sums
            # new_sums = sums U {s + vx for s in sums}
            # Optimization: if len(sums) is large, this is the bottleneck.
            new_sums = set(sums)
            for s in sums:
                new_sums.add(s + vx)
            
            # Check if any existing element's val is now a subset sum
            # Existing y in S: val[y] not in sums (old).
            # Check if val[y] in new_sums.
            # Since val[y] not in sums, it must be val[y] = s + vx for some s in sums.
            # => s = val[y] - vx.
            valid = True
            for y in S:
                vy = val[y]
                if (vy - vx) in sums:
                    valid = False
                    break
            
            if valid:
                S.append(x)
                sums = new_sums
                if len(S) > best_size:
                    best_set = list(S)
                    best_size = len(S)
                    
        return S

    # Try specific orders
    greedy(list(range(24, 0, -1)))
    greedy(list(range(1, 25)))
    
    # Random orders
    while time.time() < deadline:
        random_order = list(range(1, 25))
        random.shuffle(random_order)
        greedy(random_order)
        
        if best_size == 24:
            break
            
    return best_set
