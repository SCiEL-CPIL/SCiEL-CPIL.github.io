import time
import itertools
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # 10 seconds is plenty for this size
    
    n = 12
    points = range(n)
    
    # Generate all 5-subsets
    subsets_5 = list(itertools.combinations(points, 5))
    num_5 = len(subsets_5)
    # Map 5-subset to index
    idx_5 = {s: i for i, s in enumerate(subsets_5)}
    
    # Generate all 6-subsets
    subsets_6 = list(itertools.combinations(points, 6))
    num_6 = len(subsets_6)
    
    # Precompute mask for each 6-subset: which 5-subsets it covers
    # Using Python integers as bitmasks
    mask_6 = [0] * num_6
    # Also precompute for each 5-subset index, the list of 6-subset indices that cover it
    covers_5 = [[] for _ in range(num_5)]
    
    for j, b6 in enumerate(subsets_6):
        m = 0
        for s5 in itertools.combinations(b6, 5):
            i = idx_5[s5]
            m |= (1 << i)
            covers_5[i].append(j)
        mask_6[j] = m
        
    initial_mask = (1 << num_5) - 1
    best_blocks = []
    
    def solve(remaining_mask, current_blocks, depth):
        nonlocal best_blocks
        
        if time.time() > deadline:
            return False
            
        if remaining_mask == 0:
            if len(current_blocks) > len(best_blocks):
                best_blocks = list(current_blocks)
            return True
            
        # Find a 5-subset in remaining_mask that is covered by the fewest available 6-subsets
        # We iterate through the set bits in remaining_mask
        # Optimization: If we just pick the first one, it might be slow.
        # Let's find the one with min count.
        
        min_count = float('inf')
        min_idx = -1
        candidates_for_min = []
        
        # Iterate through all 5-subsets in remaining_mask
        # remaining_mask is an int. We can iterate through bits.
        # Since num_5 is 792, we can just loop 0..791 and check bit.
        # Or use bit manipulation to get set bits.
        r = remaining_mask
        while r:
            # Get lowest set bit
            lsb = r & -r
            i = lsb.bit_length() - 1
            r ^= lsb
            
            # Count how many 6-subsets containing i are still valid
            # A 6-subset j is valid if mask_6[j] is subset of remaining_mask
            # i.e., mask_6[j] & ~remaining_mask == 0
            
            count = 0
            for j in covers_5[i]:
                if mask_6[j] & ~remaining_mask == 0:
                    count += 1
                    if count == 1:
                        # If we find a forced move, we can break early? 
                        # No, we want the min count. If min is 1, we can't get lower.
                        # But we need to know if it's 0.
                        # If count is 0, we fail.
                        pass
            
            if count == 0:
                return False # Dead end
                
            if count < min_count:
                min_count = count
                min_idx = i
                if min_count == 1:
                    # Can't get better than 1. Break to save time?
                    # Yes, 1 is the minimum possible non-zero count.
                    break
                    
        # Now we have min_idx. We need to try all valid 6-subsets containing min_idx.
        # To improve performance, we can order them by how many "options" they leave, but simple depth-first is likely fine.
        
        for j in covers_5[min_idx]:
            if mask_6[j] & ~remaining_mask == 0:
                # Try this block
                current_blocks.append(list(subsets_6[j]))
                if solve(remaining_mask & ~mask_6[j], current_blocks, depth + 1):
                    return True
                current_blocks.pop()
                
        return False

    solve(initial_mask, [], 0)
    
    # If best_blocks is empty, it means it failed to find a solution within time or logic error.
    # But S(5,6,12) exists, so it should find it.
    # If it returns early due to deadline, best_blocks might be partial or empty.
    # The task asks to maximize. If we didn't find 132, we return what we have.
    # However, the heuristic is strong, so it should find 132 very fast.
    
    return best_blocks

# EVOLVE-BLOCK-END
