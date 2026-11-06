import sys
from itertools import combinations
import time
import numpy as np

def construct():
    """
    Returns the 132 blocks of the Steiner System S(5, 6, 12).
    Uses optimized backtracking with MRV and forward checking.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

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
            
    # Initial availability counts for each 5-subset
    # avail_count[fid] = number of blocks that can still cover fid
    avail_count = [0] * n5
    for idx in range(n6):
        for fid in six_to_fives[idx]:
            avail_count[fid] += 1
            
    remaining_6 = [True] * n6
    used_fives = [False] * n5
    blocks_found = []
    
    # Symmetry Breaking 1: Fix first block to (0,1,2,3,4,5)
    first_block_idx = all_6subsets.index((0, 1, 2, 3, 4, 5))
    blocks_found.append(list(all_6subsets[first_block_idx]))
    remaining_6[first_block_idx] = False
    for fid in six_to_fives[first_block_idx]:
        used_fives[fid] = True
        avail_count[fid] -= 1
        
    # Symmetry Breaking 2: Fix second block to the lexicographically smallest valid block
    second_block_idx = -1
    for idx in range(n6):
        if remaining_6[idx]:
            valid = True
            for fid in six_to_fives[idx]:
                if used_fives[fid]:
                    valid = False
                    break
            if valid:
                second_block_idx = idx
                break
                
    if second_block_idx != -1:
        blocks_found.append(list(all_6subsets[second_block_idx]))
        remaining_6[second_block_idx] = False
        for fid in six_to_fives[second_block_idx]:
            used_fives[fid] = True
            avail_count[fid] -= 1
            
    def backtrack():
        if time.time() > deadline:
            return False
            
        if len(blocks_found) == 132:
            return True
            
        # Find MRV: 5-subset with min available 6-subsets
        min_avail = float('inf')
        best_fid = -1
        
        for fid in range(n5):
            if used_fives[fid]:
                continue
            count = avail_count[fid]
            if count == 0:
                return False
            if count < min_avail:
                min_avail = count
                best_fid = fid
                if count == 1:
                    break
                    
        if best_fid == -1:
            # All 5-subsets covered?
            # If we haven't found 132 blocks, this shouldn't happen unless we are done
            return len(blocks_found) == 132
            
        # Get candidates for best_fid
        candidates = []
        for idx in five_to_six[best_fid]:
            if not remaining_6[idx]:
                continue
            # Check if block is valid (no other 5-subsets used)
            valid = True
            for fid in six_to_fives[idx]:
                if used_fives[fid]:
                    valid = False
                    break
            if valid:
                candidates.append(idx)
                
        if not candidates:
            return False
            
        # Sort candidates? 
        # Heuristic: Try blocks that cover more "tight" (low avail_count) subsets first?
        # Or just lexicographical?
        # Let's try sorting by the sum of avail_count of the 5-subsets in the block.
        # Lower sum means we are covering subsets that are becoming scarce.
        def score_block(idx):
            return sum(avail_count[fid] for fid in six_to_fives[idx])
            
        candidates.sort(key=score_block)
        
        for idx in candidates:
            # Apply block
            remaining_6[idx] = False
            blocks_found.append(list(all_6subsets[idx]))
            for fid in six_to_fives[idx]:
                used_fives[fid] = True
                avail_count[fid] -= 1
                
            if backtrack():
                return True
                
            # Undo
            remaining_6[idx] = True
            blocks_found.pop()
            for fid in six_to_fives[idx]:
                used_fives[fid] = False
                avail_count[fid] += 1
                
        return False

    if backtrack():
        return blocks_found
        
    return blocks_found

if __name__ == '__main__':
    res = construct()
    print(f"Number of blocks: {len(res)}")
