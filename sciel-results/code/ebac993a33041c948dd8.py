import itertools
import random
import time

def construct():
    # Precompute all 6-subsets of range(12) and their 5-subsets
    all_blocks = list(itertools.combinations(range(12), 6))
    n_blocks = len(all_blocks)
    
    # Precompute the 5-subsets for each block as a tuple of tuples for fast hashing
    block_subsets = []
    for b in all_blocks:
        subsets = tuple(sorted(itertools.combinations(b, 5)))
        block_subsets.append(subsets)
    
    target = 132
    start_time = time.time()
    time_limit = 110  # Safety margin
    
    best_blocks = []
    
    while time.time() - start_time < time_limit:
        indices = list(range(n_blocks))
        random.shuffle(indices)
        
        covered = set()
        blocks = []
        
        for idx in indices:
            subsets = block_subsets[idx]
            # Check for conflicts
            # Since subsets are tuples, we can check membership
            conflict = False
            for s in subsets:
                if s in covered:
                    conflict = True
                    break
            
            if not conflict:
                blocks.append(list(all_blocks[idx]))
                covered.update(subsets)
                
                if len(blocks) > len(best_blocks):
                    best_blocks = blocks[:]
                    
                if len(blocks) == target:
                    return blocks
        
    return best_blocks
