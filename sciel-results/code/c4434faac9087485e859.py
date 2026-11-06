import random
from math import lcm
from functools import reduce

# Calculate LCM of 1 to 22
L = reduce(lcm, range(1, 23))
# Precompute values
vals = [L // i for i in range(1, 23)]

def is_valid(S):
    """
    Checks if the subset of indices S has distinct reciprocal subset sums.
    S is a list of indices 1..22.
    """
    if not S:
        return True
    
    # Get the corresponding integer values
    v = [vals[i-1] for i in S]
    n = len(v)
    
    # Generate all subset sums
    sums = {0}
    for val in v:
        new_sums = {s + val for s in sums}
        if new_sums & sums:
            return False
        sums |= new_sums
        
    return True

def construct():
    best_size = 0
    best_set = []
    
    # 1. Deterministic greedy (ascending order)
    current_set = []
    for x in range(1, 23):
        if is_valid(current_set + [x]):
            current_set.append(x)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set[:]

    # 2. Randomized greedy
    # We can break early if we find a very large set, e.g., 12 or 13
    for _ in range(2000):
        candidates = list(range(1, 23))
        random.shuffle(candidates)
        current_set = []
        
        for x in candidates:
            if is_valid(current_set + [x]):
                current_set.append(x)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
        # Heuristic: if we reach size 12, it's likely optimal or close.
        # The problem is small, but 12 is a good target.
        if best_size >= 12:
            break

    return best_set
