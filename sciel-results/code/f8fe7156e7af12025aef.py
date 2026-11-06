import sys
from itertools import combinations
import time
import numpy as np

def construct():
    """
    Returns the 132 blocks of the Steiner System S(5, 6, 12).
    Uses optimized backtracking with bitmask state and strong symmetry breaking.
    """
    start_time = time.time()
    deadline = start_time + 100.0

    # Map 5-subsets to IDs (0..791)
    all_5subsets = list(combinations(range(12), 5))
    five_to_id = {sub: i for i, sub in enumerate(all_5subsets)}
    
    # Map 6-subsets to IDs (0..923)
    all_6subsets = list(combinations(range(12), 6))
    n6 = len(all_6subsets)  # 924
    n5 = len(all_5subsets)  # 792
    
    # Precompute: for each 6-subset, the list of 5-subset IDs it contains
    six_to_fives = []
    for block in all_6subsets:
        fives = []
        for i in range(6):
            sub = block[:i] + block[i+1:]
            fives.append(five_to_id[sub])
        six_to_fives.append(fives)
    
    # Precompute: for each 5-subset ID, the list of 6-subset indices containing it
    five_to_six = [[] for _ in range(n5)]
    for idx, fives in enumerate(six_to_fives):
        for fid in fives:
            five_to_six[fid].append(idx)
    
    # Bitmasks for speed
    # used_fives_mask: bit i is 1 if 5-subset i is used
    # remaining_6_mask: bit j is 1 if 6-subset j is available
    used_fives_mask = 0
    remaining_6_mask = (1 << n6) - 1
    
    blocks_found = []
    
    # Symmetry Breaking 1: Fix first block to (0,1,2,3,4,5)
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_idx = all_6subsets.index(first_block)
    blocks_found.append(list(first_block))
    remaining_6_mask &= ~(1 << first_block_idx)
    for fid in six_to_fives[first_block_idx]:
        used_fives_mask |= (1 << fid)
        
    # Symmetry Breaking 2: Fix second block to (0,1,2,3,6,7)
    # Verify it's valid
    second_block = (0, 1, 2, 3, 6, 7)
    second_block_idx = all_6subsets.index(second_block)
    # Check validity against first block
    valid = True
    for fid in six_to_fives[second_block_idx]:
        if (used_fives_mask >> fid) & 1:
            valid = False
            break
    if valid:
        blocks_found.append(list(second_block))
        remaining_6_mask &= ~(1 << second_block_idx)
        for fid in six_to_fives[second_block_idx]:
            used_fives_mask |= (1 << fid)
    else:
        # If not valid (shouldn't happen), find lexicographically smallest valid
        for idx in range(n6):
            if (remaining_6_mask >> idx) & 1:
                ok = True
                for fid in six_to_fives[idx]:
                    if (used_fives_mask >> fid) & 1:
                        ok = False
                        break
                if ok:
                    blocks_found.append(list(all_6subsets[idx]))
                    remaining_6_mask &= ~(1 << idx)
                    for fid in six_to_fives[idx]:
                        used_fives_mask |= (1 << fid)
                    break

    def backtrack():
        nonlocal used_fives_mask, remaining_6_mask, blocks_found
        
        if time.time() > deadline:
            return False
            
        if len(blocks_found) == 132:
            return True
            
        # MRV: Find 5-subset with fewest available 6-subsets
        # We need to iterate through unused 5-subsets
        # To speed up, we can precompute counts or just iterate.
        # 792 is small enough to iterate.
        
        min_avail = float('inf')
        best_fid = -1
        
        # Iterate through all 5-subsets
        # Optimization: Skip used ones
        # We can iterate through 0..791
        for fid in range(n5):
            if (used_fives_mask >> fid) & 1:
                continue
            
            count = 0
            for idx in five_to_six[fid]:
                if (remaining_6_mask >> idx) & 1:
                    count += 1
                    if count >= min_avail:
                        break
            
            if count == 0:
                return False
            
            if count < min_avail:
                min_avail = count
                best_fid = fid
                if count == 1:
                    break
                    
        if best_fid == -1:
            # No unused 5-subsets? Should not happen if we haven't reached 132 blocks
            # Actually, if we have placed k blocks, we have covered 6k 5-subsets.
            # If 6k < 792, there are unused 5-subsets.
            return True
            
        # Try all 6-subsets covering best_fid
        for idx in five_to_six[best_fid]:
            if not ((remaining_6_mask >> idx) & 1):
                continue
                
            # Check if block is valid (no conflict with used_fives)
            valid = True
            for fid in six_to_fives[idx]:
                if (used_fives_mask >> fid) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Apply
            remaining_6_mask &= ~(1 << idx)
            blocks_found.append(list(all_6subsets[idx]))
            for fid in six_to_fives[idx]:
                used_fives_mask |= (1 << fid)
                
            if backtrack():
                return True
                
            # Revert
            remaining_6_mask |= (1 << idx)
            blocks_found.pop()
            for fid in six_to_fives[idx]:
                used_fives_mask &= ~(1 << fid)
                
        return False

    if backtrack():
        return blocks_found
        
    return blocks_found

if __name__ == '__main__':
    res = construct()
    print(f"Number of blocks: {len(res)}")
