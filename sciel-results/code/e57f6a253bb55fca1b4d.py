import itertools
import time

def construct():
    start_time = time.time()
    
    # Vertices 0..11
    n = 12
    m = 40
    
    # Base sets: Empty and all singletons
    base_sets = [tuple()]
    for i in range(n):
        base_sets.append((i,))
        
    # We need 27 more sets
    # Strategy: 12 size-11, 10 size-10, 4 size-9
    # Generate all size-11 sets
    size_11_sets = list(itertools.combinations(range(n), 11))
    
    # Generate all size-10 sets
    size_10_sets = list(itertools.combinations(range(n), 10))
    
    # Generate all size-9 sets
    size_9_sets = list(itertools.combinations(range(n), 9))
    
    # Pick 12 size-11 (all)
    selected_11 = size_11_sets
    
    # Pick 10 size-10
    # To maximize 9-10 pairs, we should pick size-10 sets that are supersets of the selected size-9 sets.
    # Let's pick 4 size-9 sets first.
    # To maximize independence, pick 4 size-9 sets that are "far apart"?
    # Actually, any 4 will do, as long as we pick their supersets.
    selected_9 = size_9_sets[:4]
    
    # Find size-10 sets that are supersets of any selected_9
    # A size-10 set B is a superset of size-9 set A if A is a subset of B.
    # B has 10 elements. A has 9.
    # B is a superset of A if B \ A has size 1.
    # i.e., B = A U {x} for some x not in A.
    
    candidate_10s = set()
    for A in selected_9:
        A_set = set(A)
        for x in range(n):
            if x not in A_set:
                B = tuple(sorted(list(A_set) + [x]))
                candidate_10s.add(B)
                
    # We need 10 size-10 sets.
    # candidate_10s might have more than 10.
    # Pick any 10.
    selected_10 = list(candidate_10s)[:10]
    
    # If we don't have enough, fill with others
    if len(selected_10) < 10:
        all_10 = set(size_10_sets)
        for s in all_10:
            if s not in selected_10:
                selected_10.append(s)
                if len(selected_10) == 10:
                    break
                    
    # Combine
    family = base_sets + selected_11 + selected_10 + selected_9
    
    # Convert to lists of ints
    result = [list(s) for s in family]
    
    # Ensure exactly 40 distinct sets
    # base: 13
    # 11s: 12
    # 10s: 10
    # 9s: 4
    # Total: 39? 
    # 1 + 12 + 12 + 10 + 4 = 39.
    # I need 40.
    # My calculation: 1 (empty) + 12 (singles) = 13.
    # Remaining 27.
    # 12 (size 11) + 10 (size 10) + 4 (size 9) = 26.
    # I need 1 more.
    # Add 1 more size-9 set? Or size-10?
    # Let's add 1 more size-9 set.
    selected_9.append(size_9_sets[4])
    
    # Re-calculate candidate 10s?
    # The new size-9 set might need new size-10 supersets.
    # But we already picked 10 size-10 sets.
    # We can just add the 5th size-9 set.
    # It will contribute to 9-11 pairs (3 pairs).
    # It might contribute to 9-10 pairs if its supersets are in selected_10.
    
    family = base_sets + selected_11 + selected_10 + selected_9
    result = [list(s) for s in family]
    
    return result

# EVOLVE-BLOCK-END
