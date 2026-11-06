import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    all_sets = []
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            for l in range(j+1, n+1):
                for m in range(l+1, n+1):
                    mask = (1 << (i-1)) | (1 << (j-1)) | (1 << (l-1)) | (1 << (m-1))
                    all_sets.append(mask)
    
    best_family_size = 0
    best_family = []
    
    # Run multiple random restarts
    while time.time() < deadline:
        random.shuffle(all_sets)
        family = []
        # Store pairs as (mask_a, mask_b, intersection_mask)
        # To avoid O(N^2) memory growth and iteration overhead, 
        # we can just keep the list of masks in family and iterate pairs on the fly?
        # No, iterating pairs on the fly for each candidate is O(|F|^2).
        # Storing pairs allows O(1) access to K, but we still iterate all pairs.
        # Let's just store the family masks.
        
        for s in all_sets:
            # Check if s can be added
            valid = True
            # Iterate over all pairs in current family
            # Optimization: If family is small, this is fast.
            for i in range(len(family)):
                a = family[i]
                for j in range(i+1, len(family)):
                    b = family[j]
                    # Check sunflower condition for {a, b, s}
                    # a^b == a^s == b^s
                    ab = a & b
                    # If ab is not subset of s, then a^s != ab is likely, but let's check properly
                    # Actually, a^s == ab implies ab is subset of s and a^s has no other bits.
                    # So (a & s) == ab AND (b & s) == ab
                    if (a & s) == ab and (b & s) == ab:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                family.append(s)
        
        if len(family) > best_family_size:
            best_family_size = len(family)
            best_family = family[:]
            
            # If we hit the theoretical max or a high threshold, we might break early?
            # Max is 250. If we get close, keep trying to beat it.
            
    # Convert bitmasks back to sorted lists of integers
    result = []
    for mask in best_family:
        subset = []
        for i in range(1, n+1):
            if mask & (1 << (i-1)):
                subset.append(i)
        result.append(sorted(subset))
        
    return result
