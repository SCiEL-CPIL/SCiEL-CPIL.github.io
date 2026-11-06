import time
from itertools import combinations

# Precomputation
def _precompute():
    idx_map = {}
    idx = 0
    for c in combinations(range(12), 5):
        idx_map[c] = idx
        idx += 1
        
    block_masks = {}
    for b in combinations(range(12), 6):
        mask = 0
        for f in combinations(b, 5):
            mask |= (1 << idx_map[f])
        block_masks[b] = mask
        
    return block_masks

BLOCK_MASKS = _precompute()
ALL_BLOCKS = list(BLOCK_MASKS.keys())
FIRST_BLOCK = (0, 1, 2, 3, 4, 5)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    covered_mask = 0
    blocks = []
    
    # Add first block
    blocks.append(list(FIRST_BLOCK))
    covered_mask |= BLOCK_MASKS[FIRST_BLOCK]
    
    # Recursive search
    # We use a list to hold the current path for backtracking
    # To avoid recursion limit issues (depth 132 is fine), we use recursion.
    
    def search(current_blocks, current_covered, depth):
        if time.time() > deadline:
            return False
        if depth == 132:
            return True
            
        # Try all blocks
        # Optimization: We can sort candidates? 
        # For now, just iterate.
        for b in ALL_BLOCKS:
            if b == FIRST_BLOCK:
                continue
                
            mask = BLOCK_MASKS[b]
            if mask & current_covered:
                continue
                
            # Try this block
            new_covered = current_covered | mask
            current_blocks.append(list(b))
            
            if search(current_blocks, new_covered, depth + 1):
                return True
                
            current_blocks.pop()
            
        return False
        
    # Start search
    # We need to pass the state. 
    # Note: The first block is already added. So we are looking for 131 more.
    # But the search function adds to `blocks`.
    # Let's restructure slightly to be cleaner.
    
    # The above `search` modifies `blocks` and `current_covered` passed by value (int) and reference (list).
    # This works.
    
    found = search(blocks, covered_mask, 1)
    
    # If not found 132, we might have a partial list.
    # But with the first block fixed, it should find the Witt design.
    
    return blocks
