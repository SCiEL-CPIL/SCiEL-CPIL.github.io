"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    best_set = []
    best_size = 0
    start_time = time.time()
    
    # Helper to check if adding e to a set with sums old_sums is valid
    # old_sums is a set of non-empty subset sums of the current elements
    # New sums will be old_sums U {e} U {s+e for s in old_sums}
    # We need to ensure:
    # 1. All new sums are distinct (implied by unique subset sums if we check collisions)
    #    Actually, we just check if len(new_sums) == 2*len(old_sums) + 1
    # 2. No two distinct sums in new_sums divide each other.
    
    def is_valid_addition(old_sums, e):
        # Generate the new sums that are NOT in old_sums
        new_only = set()
        new_only.add(e)
        for s in old_sums:
            ns = s + e
            # If ns is in old_sums, it's a collision (subset sum not unique)
            # We don't explicitly check collision here, but we can check size later.
            # However, if ns is in old_sums, it means two subsets have same sum.
            # The problem states: "Two different subsets with the same sum still make the set invalid"
            # So we MUST ensure uniqueness.
            if ns in old_sums:
                return False
            new_only.add(ns)
            
        # Check if e is in old_sums (collision with single element)
        if e in old_sums:
            return False
            
        # Now we have new_only sums.
        # We need to check divisibility between:
        # 1. Pairs within new_only
        # 2. Pairs between new_only and old_sums
        # (Pairs within old_sums are already valid)
        
        # 1. Check new_only internal
        new_list = sorted(new_only)
        for i in range(len(new_list)):
            si = new_list[i]
            for j in range(i + 1, len(new_list)):
                sj = new_list[j]
                if sj % si == 0:
                    return False
                    
        # 2. Check new_only vs old_sums
        for sn in new_only:
            for so in old_sums:
                if sn < so:
                    if so % sn == 0:
                        return False
                else:
                    if sn % so == 0:
                        return False
                        
        return True

    # Randomized Greedy Search with multiple restarts
    # We try to build a set by picking candidates in a random order.
    # We repeat this many times to find a good set.
    
    num_restarts = 500
    for r in range(num_restarts):
        if time.time() - start_time > 55:
            break
            
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        current = []
        current_sums = set()
        
        for c in candidates:
            # Quick prune: if we can't beat best_size even if we add all remaining, skip?
            # len(current) + (len(candidates) - index) is an upper bound, but weak.
            
            if is_valid_addition(current_sums, c):
                current.append(c)
                # Update current_sums
                # New sums are old_sums U {c} U {s+c for s in old_sums}
                new_sums = set(current_sums)
                new_sums.add(c)
                for s in current_sums:
                    new_sums.add(s + c)
                current_sums = new_sums
                
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
