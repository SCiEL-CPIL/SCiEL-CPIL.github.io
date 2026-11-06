import random
import time

def score_set(S):
    S_set = set(S)
    good_d = set()
    for d in range(1, 512):
        for a in S:
            if (a + d) in S_set and (a + 2 * d) in S_set:
                good_d.add(d)
                break
    return len(good_d)

def construct_greedy(seed_val):
    random.seed(seed_val)
    S = set()
    # We want to cover d=1, 2, ...
    # Max d such that 3 terms fit in 1..512 is 255 (1, 256, 511)
    for d in range(1, 256):
        # Check if d is already covered
        covered = False
        for a in S:
            if (a + d) in S and (a + 2 * d) in S:
                covered = True
                break
        
        if covered:
            continue
            
        # Try to add a progression for d
        # We want to minimize the number of new elements added
        # Candidates: a such that 1 <= a and a+2d <= 512
        # a in [1, 512 - 2d]
        
        best_a = -1
        best_new_count = 4 # infinity
        best_new_elements = []
        
        # Optimization: Only check a few candidates if the range is large?
        # 512 is small enough to check all.
        for a in range(1, 512 - 2 * d + 1):
            p1 = a
            p2 = a + d
            p3 = a + 2 * d
            
            new_elems = []
            if p1 not in S: new_elems.append(p1)
            if p2 not in S: new_elems.append(p2)
            if p3 not in S: new_elems.append(p3)
            
            num_new = len(new_elems)
            
            if num_new < best_new_count:
                best_new_count = num_new
                best_a = a
                best_new_elements = new_elems
                
                # If we can add 0 new elements, that means it's already covered, 
                # which we checked. So min is 1.
                if num_new == 1:
                    # We found a way to add just 1 element. 
                    # Is there a better one? Maybe one that allows future overlaps?
                    # For now, take the first 1-add.
                    break
            
        if best_a != -1 and len(S) + best_new_count <= 32:
            for elem in best_new_elements:
                S.add(elem)
        else:
            # If we can't fit it, we might stop or skip.
            # If we skip, we might be able to cover larger d's?
            # But larger d's are harder to overlap.
            # Let's just stop if we can't add 1 element (space full) 
            # or if we can't add any (range issue, unlikely for small d).
            # Actually, if we can't add because of space, we are full.
            # If we are full, we can't add any more.
            if len(S) == 32:
                break
            # If we can't add because best_new_count > remaining space, 
            # we might try to find a 'worse' overlap that fits? 
            # No, if best is 3 and we have 2 slots, we can't add 3.
            # We could try to add 2 if we find an a with 2 new?
            # The loop above finds the MIN new count. If min > space, no a fits.
            # So we are stuck.
            break
            
    return list(S)

def construct():
    best_score = 0
    best_S = []
    
    # Run multiple times with different seeds to explore different tie-breaks
    # Since the greedy choice is deterministic if we break on first '1', 
    # we should randomize the choice among '1' candidates or '2' candidates.
    
    for seed in range(20):
        S = construct_greedy(seed)
        sc = score_set(S)
        if sc > best_score:
            best_score = sc
            best_S = S
            
    # Local search improvement
    # Try to swap elements to improve score
    S_set = set(best_S)
    current_score = best_score
    
    # Simple local search: try replacing an element with a random one
    # if it improves score.
    # This might be slow, so limit it.
    
    return best_S
