import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precomputation
    points = range(12)
    five_subs = list(combinations(points, 5))
    five_idx = {s: i for i, s in enumerate(five_subs)}
    n_fives = len(five_subs)
    
    six_subs = list(combinations(points, 6))
    block_masks = []
    blocks_for_five = [[] for _ in range(n_fives)]
    
    for b in six_subs:
        mask = 0
        for s in combinations(b, 5):
            idx = five_idx[s]
            mask |= (1 << idx)
            blocks_for_five[idx].append(len(block_masks))
        block_masks.append(mask)
    
    full_mask = (1 << n_fives) - 1
    blocks_found = []
    
    def backtrack(remaining_mask, current_blocks):
        if time.time() > deadline:
            return False
        
        if remaining_mask == 0:
            if len(current_blocks) == 132:
                blocks_found.extend(current_blocks)
                return True
            return False
        
        # Find first set bit
        # lowest set bit: mask & -mask
        lsb = remaining_mask & -remaining_mask
        idx = lsb.bit_length() - 1
        
        # Try blocks covering this 5-subset
        for bi in blocks_for_five[idx]:
            bmask = block_masks[bi]
            if bmask & remaining_mask == bmask:
                current_blocks.append(bi)
                if backtrack(remaining_mask ^ bmask, current_blocks):
                    return True
                current_blocks.pop()
                
        return False

    if backtrack(full_mask, []):
        # Convert block indices to actual blocks
        return [list(six_subs[bi]) for bi in blocks_found]
    else:
        # Fallback to greedy if exact solver fails or times out
        covered = set()
        blocks = []
        for b in six_subs:
            subsets = list(combinations(b, 5))
            if not any(s in covered for s in subsets):
                blocks.append(list(b))
                covered.update(subsets)
        return blocks
