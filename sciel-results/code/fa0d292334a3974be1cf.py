import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Points for S(4,5,11) are 1..11
    points_11 = list(range(1, 12))
    
    # Generate all 4-subsets and 5-subsets of points_11
    all_4_subsets = list(combinations(points_11, 4))
    all_5_subsets = list(combinations(points_11, 5))
    
    # Precompute 4-subsets for each 5-subset
    subsets_4_of_5 = [list(combinations(b, 4)) for b in all_5_subsets]
    
    # Map 4-subset -> list of indices of 5-subsets containing it
    map_4_to_5_indices = {s: [] for s in all_4_subsets}
    for i, b in enumerate(all_5_subsets):
        for s in subsets_4_of_5[i]:
            map_4_to_5_indices[s].append(i)
            
    # Initial state
    # We need to cover all 4-subsets.
    # Start with empty covered, all uncovered.
    all_4_set = set(all_4_subsets)
    
    def solve(covered, uncovered, current_blocks):
        if time.time() > deadline:
            return None
            
        if len(current_blocks) == 66:
            return current_blocks
            
        if not uncovered:
            # Should not happen before 66 blocks in a valid S(4,5,11)
            return current_blocks if len(current_blocks) == 66 else None
            
        # Heuristic: Pick uncovered 4-subset with fewest valid candidates
        min_count = float('inf')
        target_subset = None
        valid_candidates = []
        
        # To speed up, we can iterate through uncovered and check counts
        # But iterating through all 330 might be slow if done naively in Python loops many times.
        # However, the search tree is small.
        
        # Optimization: Instead of checking all uncovered, we can just pick the first one?
        # No, MRV is crucial for speed.
        
        # Let's iterate through uncovered
        for s in uncovered:
            # Count valid candidates for s
            count = 0
            for idx in map_4_to_5_indices[s]:
                # Check if candidate idx is compatible with covered
                # i.e., no 4-subset of all_5_subsets[idx] is in covered
                # Note: s is in uncovered, so s is NOT in covered.
                # We only need to check the OTHER 4 subsets of the candidate.
                # Actually, it's easier to just check all 5.
                compatible = True
                for sub in subsets_4_of_5[idx]:
                    if sub in covered:
                        compatible = False
                        break
                if compatible:
                    count += 1
                    if count > min_count:
                        break # Found a better candidate (smaller count)
                        
            if count < min_count:
                min_count = count
                target_subset = s
                valid_candidates = []
                # Re-collect valid candidates for this target
                for idx in map_4_to_5_indices[s]:
                    compatible = True
                    for sub in subsets_4_of_5[idx]:
                        if sub in covered:
                            compatible = False
                            break
                    if compatible:
                        valid_candidates.append(idx)
                        
            if min_count == 0:
                return None # Dead end
                
        # If no uncovered subsets found? (Should be handled by 'if not uncovered')
        if target_subset is None:
            return None
            
        # Try each valid candidate
        for idx in valid_candidates:
            b = all_5_subsets[idx]
            new_covered = set(covered)
            for s in subsets_4_of_5[idx]:
                new_covered.add(s)
            new_uncovered = uncovered - new_covered
            
            result = solve(new_covered, new_uncovered, current_blocks + [b])
            if result is not None:
                return result
                
        return None

    # Start solving S(4,5,11)
    # To help the solver, we can fix the first block to break symmetry.
    # E.g., fix block (1,2,3,4,5)
    first_block = (1, 2, 3, 4, 5)
    if first_block in all_5_subsets:
        idx_first = all_5_subsets.index(first_block)
        init_covered = set(subsets_4_of_5[idx_first])
        init_uncovered = all_4_set - init_covered
        init_blocks = [first_block]
        sol_4511 = solve(init_covered, init_uncovered, init_blocks)
    else:
        sol_4511 = solve(set(), all_4_set, [])
        
    if sol_4511 is None:
        # Fallback: Return empty or a greedy attempt?
        # Given the small size, this shouldn't happen.
        # But to be safe, return []
        return []
        
    # Construct S(5,6,12) from S(4,5,11)
    # Points 0..11
    # For each block B in sol_4511 (subset of 1..11):
    #   Add {0} U B
    #   Add (1..11) \ B
    
    final_blocks = []
    for b in sol_4511:
        # b is a tuple of 5 elements from 1..11
        block_with_0 = [0] + list(b)
        final_blocks.append(block_with_0)
        
        # Complement in 1..11
        b_set = set(b)
        comp = [p for p in points_11 if p not in b_set]
        final_blocks.append(comp)
        
    return final_blocks
