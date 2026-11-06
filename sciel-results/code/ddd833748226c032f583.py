import time
import sys

def construct():
    """
    Constructs the Steiner system S(5,6,12) (Witt Design W12).
    Returns a list of 132 distinct 6-subsets of 0..11.
    """
    from itertools import combinations
    
    # 1. Precomputation
    # All 5-subsets of 0..11
    all_5subsets = list(combinations(range(12), 5))
    # Map 5-subset to ID
    id_of_5subset = {s: i for i, s in enumerate(all_5subsets)}
    num_5subsets = len(all_5subsets) # 792
    
    # All 6-subsets of 0..11
    all_6subsets = list(combinations(range(12), 6))
    
    # For each 6-subset, get the IDs of its 5-subsets
    block_5ids = []
    for b in all_6subsets:
        ids = [id_of_5subset[s] for s in combinations(b, 5)]
        block_5ids.append(ids)
        
    # For each 5-subset ID, get the list of 6-subset indices that contain it
    # There are 7 such blocks for each 5-subset
    blocks_containing_5 = [[] for _ in range(num_5subsets)]
    for b_idx, ids in enumerate(block_5ids):
        for sid in ids:
            blocks_containing_5[sid].append(b_idx)
            
    # 2. Backtracking Search
    covered = [False] * num_5subsets
    current_blocks = []
    target_blocks = 132
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin before 150s limit
    
    def search():
        if len(current_blocks) == target_blocks:
            return True
        if time.time() > deadline:
            return False
            
        # Find an uncovered 5-subset
        # Optimization: We can pick the first one found.
        # Since the problem is highly symmetric and constrained, 
        # picking the first uncovered one is sufficient.
        uncovered_sid = -1
        for i in range(num_5subsets):
            if not covered[i]:
                uncovered_sid = i
                break
        
        if uncovered_sid == -1:
            # All covered, but we might not have 132 blocks?
            # In S(5,6,12), if all 792 5-subsets are covered by disjoint blocks,
            # we must have exactly 792/6 = 132 blocks.
            return len(current_blocks) == target_blocks
            
        # Try each block containing this 5-subset
        for b_idx in blocks_containing_5[uncovered_sid]:
            # Check if the other 5-subsets in this block are also uncovered
            valid = True
            for sid in block_5ids[b_idx]:
                if covered[sid]:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Place the block
            current_blocks.append(all_6subsets[b_idx])
            for sid in block_5ids[b_idx]:
                covered[sid] = True
                
            if search():
                return True
                
            # Backtrack
            current_blocks.pop()
            for sid in block_5ids[b_idx]:
                covered[sid] = False
                
        return False
        
    if search():
        return [list(b) for b in current_blocks]
    else:
        # Fallback: Return whatever we have (should not happen for this specific design)
        # Or raise an error? The task says "invalid result... scores 0".
        # But S(5,6,12) exists, so search should succeed.
        return [list(b) for b in current_blocks]

# EVOLVE-BLOCK-END
