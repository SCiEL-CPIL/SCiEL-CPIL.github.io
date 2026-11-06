import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # 1. Map 5-subsets to IDs
    # There are C(12, 5) = 792 5-subsets
    # We need a fast way to get the ID of a 5-subset.
    # We can precompute a dictionary: frozenset(5_subset) -> id
    
    points = range(12)
    all_5_subsets = [frozenset(s) for s in combinations(points, 5)]
    subset_to_id = {s: i for i, s in enumerate(all_5_subsets)}
    
    # 2. Generate all 6-subsets (blocks)
    all_6_subsets = [tuple(s) for s in combinations(points, 6)]
    
    # 3. Precompute masks for each 6-subset
    # Mask is an integer where bit i is set if the 5-subset with id i is contained in the block.
    block_masks = []
    for block in all_6_subsets:
        mask = 0
        for i in range(6):
            # Remove block[i], the remaining 5 form a 5-subset
            # This is inefficient. Better: iterate combinations of block of size 5.
            pass
        
        # Efficient way:
        # A block has 6 points. It contains 6 5-subsets.
        # Each 5-subset is formed by removing one point from the block.
        for i in range(6):
            # Create the 5-subset by excluding block[i]
            # We need the frozenset of the other 5 points
            # To make it fast, we can just construct the frozenset
            # But we need to map it to ID.
            # Let's precompute the 5-subsets for each 6-subset once.
            pass

    # Let's redo precomputation properly.
    # For each 6-subset, find the 6 5-subsets it contains.
    # Store their IDs in a list.
    # Then create the mask.
    
    # Also, for each 5-subset ID, store the list of 6-subset indices that contain it.
    # candidates_for_5 = [ [] for _ in range(792) ]
    
    # Optimization: 
    # We will fix the first block to be (0, 1, 2, 3, 4, 5).
    # This reduces the search space significantly.
    
    # Let's build the structures.
    
    # Map 5-subset to ID
    # Using a dictionary of frozensets is okay for 792 items.
    
    # Precompute for each 6-subset:
    # 1. The mask (int)
    # 2. The list of 5-subset IDs it covers
    
    # And for each 5-subset ID:
    # 3. The list of 6-subset indices that cover it
    
    block_5_ids = [] # list of lists of 5-subset IDs
    block_masks = [] # list of ints
    
    for b in all_6_subsets:
        ids = []
        mask = 0
        for i in range(6):
            # Form 5-subset by removing b[i]
            # b is a tuple of 6 ints
            # Create a set of the other 5
            # To avoid set creation overhead in the loop, we can do this:
            # Since b is sorted, we can create the frozenset efficiently?
            # Actually, 924 * 6 = 5544 set creations is trivial.
            s = frozenset(b[:i] + b[i+1:])
            id_ = subset_to_id[s]
            ids.append(id_)
            mask |= (1 << id_)
        block_5_ids.append(ids)
        block_masks.append(mask)
        
    # candidates_for_5: for each 5-subset ID, which blocks contain it?
    candidates_for_5 = [[] for _ in range(792)]
    for b_idx, ids in enumerate(block_5_ids):
        for id_ in ids:
            candidates_for_5[id_].append(b_idx)
            
    ALL_MASK = (1 << 792) - 1
    
    # Fix first block: (0, 1, 2, 3, 4, 5)
    # Find its index
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_idx = all_6_subsets.index(first_block)
    
    initial_mask = block_masks[first_block_idx]
    initial_blocks = [list(first_block)]
    
    # Backtracking search
    # We need to find the remaining 131 blocks.
    # To speed up, we can use a recursive function.
    # Python recursion limit might be an issue? Depth 132 is fine.
    
    def backtrack(mask, blocks):
        if time.time() > deadline:
            return None
            
        if mask == ALL_MASK:
            return blocks
        
        # MRV heuristic: find uncovered 5-subset with fewest valid candidates
        best_uncovered = -1
        best_candidates = []
        best_count = float('inf')
        
        for id_ in range(792):
            if not (mask & (1 << id_)):
                cands = candidates_for_5[id_]
                valid_cands = [b for b in cands if not (block_masks[b] & mask)]
                count = len(valid_cands)
                if count == 0:
                    return None  # Dead end
                if count < best_count:
                    best_count = count
                    best_uncovered = id_
                    best_candidates = valid_cands
                    if count == 1:
                        break  # Can't do better than 1
        
        if best_uncovered == -1:
            return blocks
        
        # Order candidates by how many new 5-subsets they cover (more first)
        best_candidates.sort(key=lambda b: -bin(block_masks[b] & ~mask).count('1'))
        
        for b_idx in best_candidates:
            b_mask = block_masks[b_idx]
            new_mask = mask | b_mask
            new_blocks = blocks + [list(all_6_subsets[b_idx])]
            
            result = backtrack(new_mask, new_blocks)
            if result is not None:
                return result
                
        return None

    result = backtrack(initial_mask, initial_blocks)
    
    if result is not None:
        return result
    else:
        # Fallback: return whatever we have? Or empty?
        # The task implies we should maximize. If we fail, we return best effort.
        # But with symmetry breaking, it should succeed.
        return initial_blocks
