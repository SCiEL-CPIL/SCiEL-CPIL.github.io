import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Generate all 4-subsets of [1, 16] as bitmasks
    # Elements are 0..15, so bits 0..15
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
        
    n_sets = len(all_sets)
    
    def is_valid_addition(family, new_set):
        """
        Check if adding new_set to family creates any sunflower.
        A sunflower is formed by A, B, C in family U {new_set} such that
        A & B == A & C == B & C.
        Since family is already valid, we only need to check triples involving new_set.
        So, for all pairs A, B in family, check if A & B == A & new_set == B & new_set.
        """
        # Optimization: If new_set is disjoint from many, etc.
        # Just iterate pairs.
        f_len = len(family)
        for i in range(f_len):
            a = family[i]
            an = a & new_set
            for j in range(i + 1, f_len):
                b = family[j]
                # Check if a & b == an and b & new_set == an
                # Note: a & b == an implies a & b & new_set == an, etc.
                if (a & b) == an and (b & new_set) == an:
                    return False
        return True

    best_family = []
    
    # We will run multiple random restarts
    # Each restart: shuffle all_sets, greedily pick valid sets
    # To improve, we can also try to add sets from the "leftovers" in random order at the end
    
    max_restarts = 100 # Adjust based on time
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        random.shuffle(all_sets)
        family = []
        
        for s in all_sets:
            if time.time() > deadline:
                break
            if is_valid_addition(family, s):
                family.append(s)
                
        # Local Search / Polish: Try to add any remaining sets
        # This is often not needed if we shuffled, but let's try a second pass
        # Actually, the greedy pass already iterated all sets. 
        # A second pass with a different order might find sets that were skipped 
        # because of earlier choices? No, greedy is order dependent. 
        # If we skipped S because of A,B, and later we still have A,B, S is still invalid.
        # So a second pass over the same family won't add anything unless the family changed.
        
        if len(family) > len(best_family):
            best_family = family[:]
            
        # If we have time, try to improve the current best_family via swaps?
        # This is complex. Let's stick to multiple restarts.
        
    # Convert bitmasks back to sorted lists of 1-based indices
    result = []
    for mask in best_family:
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result
