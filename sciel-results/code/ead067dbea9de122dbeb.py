import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    # Allow 110s for total search to be safe
    deadline = start_time + 110.0
    
    points = range(12)
    all_5_subsets = list(combinations(points, 5))
    index_5 = {s: i for i, s in enumerate(all_5_subsets)}
    
    all_6_subsets = list(combinations(points, 6))
    num_6 = len(all_6_subsets)
    
    block_masks = [0] * num_6
    block_5_indices = [None] * num_6
    
    for j, b in enumerate(all_6_subsets):
        mask = 0
        inds = []
        for s in combinations(b, 5):
            idx = index_5[s]
            mask |= (1 << idx)
            inds.append(idx)
        block_masks[j] = mask
        block_5_indices[j] = inds

    subsets_5_to_6 = [[] for _ in range(792)]
    for i in range(792):
        s5 = all_5_subsets[i]
        s5_set = set(s5)
        for j, b in enumerate(all_6_subsets):
            if s5_set.issubset(set(b)):
                subsets_5_to_6[i].append(j)

    # We will try multiple random restarts
    # Each restart tries to find a full 132-block packing
    # We fix the first block to break symmetry, but we can vary WHICH block containing 0 we pick, 
    # or just randomize the order of subsequent choices.
    # Actually, fixing (0,1,2,3,4,5) is fine, but randomizing the rest is key.
    
    best_result = []
    
    # Try a few different seeds for randomization
    for seed in range(5):
        if time.time() > deadline - 10: # Reserve 10s for final return
            break
            
        rng = random.Random(seed)
        
        # We can also randomize the initial block choice among those containing 0
        # But (0,1,2,3,4,5) is a good canonical start. 
        # Let's stick to that to reduce search space, but randomize the rest.
        
        first_block = (0, 1, 2, 3, 4, 5)
        first_block_idx = all_6_subsets.index(first_block)
        
        blocks = [first_block]
        covered_mask = block_masks[first_block_idx]
        uncovered = [i for i in range(792) if not (covered_mask >> i) & 1]
        
        def backtrack(covered_mask, uncovered, depth):
            if len(blocks) == 132:
                return True
            
            if time.time() > deadline - 10:
                return False
                
            if not uncovered:
                return False
            
            # MRV Heuristic
            min_candidates = float('inf')
            target_idx = -1
            valid_cands = []
            
            # Iterate through uncovered to find the one with fewest valid blocks
            # Optimization: We can sample or just scan. Scanning 792 items is fast.
            for i in uncovered:
                cands = subsets_5_to_6[i]
                valid = []
                for j in cands:
                    if (covered_mask & block_masks[j]) == 0:
                        valid.append(j)
                
                if len(valid) < min_candidates:
                    min_candidates = len(valid)
                    target_idx = i
                    valid_cands = valid
                    if min_candidates == 0:
                        break

            if target_idx == -1 or min_candidates == 0:
                return False

            # Randomize the order of candidates to escape deterministic traps
            rng.shuffle(valid_cands)
            
            for j in valid_cands:
                blocks.append(all_6_subsets[j])
                
                new_mask = block_masks[j]
                new_covered_mask = covered_mask | new_mask
                # Filter uncovered list
                # Optimization: Using list comprehension is okay, but we can be smarter if needed.
                # For now, keep it simple as 792 is small.
                new_uncovered = [k for k in uncovered if not (new_mask >> k) & 1]
                
                if backtrack(new_covered_mask, new_uncovered, depth + 1):
                    return True
                
                blocks.pop()
            
            return False

        if backtrack(covered_mask, uncovered, 1):
            # Found a solution
            best_result = [list(b) for b in blocks]
            break # Success, no need to try more seeds
        
        # If it didn't find 132, we might have a partial solution in `blocks`
        # We should keep the best partial solution found so far
        if len(blocks) > len(best_result):
            best_result = [list(b) for b in blocks]
            # Reset blocks for next iteration? 
            # The `blocks` list is local to the seed loop scope? No, it's defined inside.
            # Wait, `blocks` is defined inside the `for seed` loop.
            # So `best_result` update is correct.

    return best_result
