import time
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0
    
    points = range(12)
    
    # Precompute all 5-subsets and 6-subsets
    all_5_subsets = list(combinations(points, 5))
    all_6_subsets = list(combinations(points, 6))
    
    # For each 6-subset, precompute its 5-subsets
    block_to_5s = {}
    for b in all_6_subsets:
        block_to_5s[b] = list(combinations(b, 5))
    
    # For each 5-subset, precompute which 6-subsets contain it
    # This is for potential use, but we'll mainly use the reverse: 
    # given a 5-subset, find 6-subsets containing it.
    # Actually, let's precompute: for each 5-subset, the list of 6-subsets that contain it.
    # There are 792 5-subsets. Each 5-subset is contained in 12-5=7 6-subsets.
    # So we can build a map: 5-subset -> list of 6-subsets containing it.
    five_to_sixes = {s: [] for s in all_5_subsets}
    for b in all_6_subsets:
        for s in block_to_5s[b]:
            five_to_sixes[s].append(b)
    
    covered = set()
    blocks = []
    
    # Symmetry breaking:
    # The first block must contain {0,1,2,3,4}. 
    # By symmetry, we can assume the 6th element is 5.
    # So Block 0 is {0,1,2,3,4,5}.
    
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_5s = block_to_5s[first_block]
    
    blocks.append(first_block)
    covered.update(first_block_5s)
    
    # Now we need 131 more blocks.
    # We will search for the next block.
    
    def backtrack():
        if len(blocks) == 132:
            return True
        
        if time.time() > deadline:
            return False
            
        # Find the first uncovered 5-subset in lexicographic order
        # This ensures we fill gaps systematically.
        # However, iterating all 792 5-subsets each step is $132 \times 792 \approx 100k$ ops, which is fast.
        
        target_5 = None
        for s in all_5_subsets:
            if s not in covered:
                target_5 = s
                break
                
        if target_5 is None:
            # All covered, but we don't have 132 blocks?
            # This shouldn't happen if we stop at 132.
            return False
            
        # Find all 6-subsets that contain target_5
        # Using precomputed map for speed
        candidates = five_to_sixes[target_5]
        
        # Filter candidates: no overlap in 5-subsets
        valid_candidates = []
        for b in candidates:
            # Check if this block is valid (no overlap in 5-subsets)
            block_5s = block_to_5s[b]
            if not any(s in covered for s in block_5s):
                valid_candidates.append(b)
                
        # Try each candidate
        for b in valid_candidates:
            # Add block
            blocks.append(b)
            block_5s = block_to_5s[b]
            covered.update(block_5s)
            
            if backtrack():
                return True
                
            # Undo
            blocks.pop()
            covered.difference_update(block_5s)
            
        return False
        
    if backtrack():
        return [list(b) for b in blocks]
    else:
        # Fallback: Return what we have or a greedy result?
        # If backtrack fails, it might be due to time or logic error.
        # But S(5,6,12) exists, so it should find it.
        # If it times out, we return the partial list?
        # The task says "Maximize the number of blocks".
        # Returning a partial list is better than nothing.
        return [list(b) for b in blocks]
