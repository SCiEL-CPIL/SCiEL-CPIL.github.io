import random
import time
from math import gcd

def lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    best_size = 0
    best_set = []
    
    candidates = list(range(1, 23))
    
    # Try a few deterministic good orders first?
    # Powers of 2: 1, 2, 4, 8, 16
    # Then fill in others.
    
    for _ in range(500): # Reduced restarts
        if time.time() > deadline:
            break
            
        order = candidates[:]
        random.shuffle(order)
        
        current_set = []
        sums = {0}
        denom = 1
        remaining = set(candidates)
        
        for x in order:
            remaining.discard(x)
            
            # Pruning
            if len(current_set) + len(remaining) <= best_size:
                break
            
            new_denom = lcm(denom, x)
            scale = new_denom // denom
            add_val = new_denom // x
            
            new_sums = set()
            is_valid = True
            
            for s in sums:
                s_scaled = s * scale
                v1 = s_scaled
                v2 = s_scaled + add_val
                v3 = s_scaled - add_val
                
                if v1 in new_sums or v2 in new_sums or v3 in new_sums:
                    is_valid = False
                    break
                new_sums.add(v1)
                new_sums.add(v2)
                new_sums.add(v3)
                
            if is_valid:
                sums = new_sums
                denom = new_denom
                current_set.append(x)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    return best_set
