import time
import random
from itertools import combinations

# Precompute all 4-element subsets of [1, 16] as bitmasks
ALL_SETS = []
for c in combinations(range(1, 17), 4):
    mask = 0
    for x in c:
        mask |= (1 << (x - 1))
    ALL_SETS.append(mask)

def can_add(family, c):
    n = len(family)
    for i in range(n):
        a = family[i]
        for j in range(i + 1, n):
            b = family[j]
            # Check sunflower condition: a&b == a&c == b&c
            if (a & b) == (a & c) == (b & c):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_family = []
    best_size = 0
    
    # We can also try a few specific orders if random fails to beat seed quickly?
    # But random is generally better.
    
    while time.time() < deadline:
        candidates = ALL_SETS[:]
        random.shuffle(candidates)
        
        current_family = []
        
        for s in candidates:
            if can_add(current_family, s):
                current_family.append(s)
                
        if len(current_family) > best_size:
            best_size = len(current_family)
            best_family = current_family[:]
            
    result = []
    for mask in best_family:
        elements = [i + 1 for i in range(16) if mask & (1 << i)]
        result.append(elements)
        
    return result
