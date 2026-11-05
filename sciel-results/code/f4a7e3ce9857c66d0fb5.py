import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    # Precompute all 6-subsets
    all_blocks = list(combinations(range(12), 6))
    n_blocks = len(all_blocks)
    
    # Precompute 5-subsets for each block as frozensets for fast hashing
    block_5subs = []
    for b in all_blocks:
        block_5subs.append(frozenset(combinations(b, 5)))
        
    # Map block to its index for quick reference if needed, 
    # but we will just iterate over indices.
    
    best_len = 0
    best_result = []
    
    # We will run multiple restarts.
    # Each restart uses a different random shuffle.
    
    for restart in range(50000):
        if time.time() > deadline:
            break
            
        # Fix the first block to {0,1,2,3,4,5} to reduce symmetry.
        # Index of (0,1,2,3,4,5) in all_blocks
        # all_blocks is lexicographic. (0,1,2,3,4,5) is the first one.
        first_block_idx = 0
        first_block = all_blocks[first_block_idx]
        
        blocks = [first_block]
        covered = set(block_5subs[first_block_idx])
        
        # Create a shuffled list of indices for the remaining blocks
        remaining_indices = list(range(1, n_blocks))
        random.shuffle(remaining_indices)
        
        # Greedily add blocks
        for idx in remaining_indices:
            if len(blocks) == 132:
                break
            if time.time() > deadline:
                break
                
            b_5subs = block_5subs[idx]
            # Check if this block is valid
            # A block is valid if none of its 5-subsets are in covered
            # Since frozensets are hashable, this is fast.
            if not any(s in covered for s in b_5subs):
                blocks.append(all_blocks[idx])
                covered.update(b_5subs)
                
        if len(blocks) > best_len:
            best_len = len(blocks)
            best_result = [list(b) for b in blocks]
            if best_len == 132:
                return best_result
                
    return best_result
