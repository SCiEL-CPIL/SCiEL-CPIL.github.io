# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    # Generate all 4-element subsets of [1,16] as bitmasks
    # Elements 1..16 correspond to bits 0..15
    all_sets = []
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        all_sets.append(mask)
    
    best_family = []
    best_size = 0
    
    # We will try multiple random orderings.
    # The sunflower check is O(N^2) per insertion.
    # With bitmasks, each check is very fast.
    # Let's estimate: 250 insertions * (250^2 / 2) pairs * ~2ns/op = ~15ms per run? 
    # Actually Python loops are slower. ~10-50ms per run is reasonable.
    # So we can do ~1000-5000 runs in 85s.
    
    # To ensure we get a good result, we'll do a mix of random and some structured attempts.
    
    for attempt in range(2000):
        if time.time() > deadline:
            break
            
        if attempt == 0:
            # Lexicographic order (which is just the order of combinations)
            candidates = list(all_sets)
        else:
            # Random order
            candidates = list(all_sets)
            random.shuffle(candidates)
        
        family = []
        # We store masks in family for fast access
        for s in candidates:
            # Check if s forms a sunflower with any pair in family
            is_valid = True
            n = len(family)
            for i in range(n):
                a = family[i]
                for j in range(i + 1, n):
                    b = family[j]
                    ab = a & b
                    # Check if a & s == ab AND b & s == ab
                    if (a & s) == ab and (b & s) == ab:
                        is_valid = False
                        break
                if not is_valid:
                    break
            
            if is_valid:
                family.append(s)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
        # If we hit the theoretical max or a very high number, we might stop early?
        # The max is 250. If we reach 250, we can stop.
        if best_size >= 250:
            break

    # Convert best_family masks back to lists of sorted integers
    result = []
    for mask in best_family:
        elems = []
        for k in range(16):
            if mask & (1 << k):
                elems.append(k + 1)
        result.append(elems)
        
    return result
# EVOLVE-BLOCK-END
