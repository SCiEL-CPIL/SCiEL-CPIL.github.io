import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Precompute triple IDs and info
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    num_words = (num_triples + 63) // 64
    
    # Map triple to (word_idx, bit_mask)
    triple_info_map = {}
    for i, triple in enumerate(triples):
        w = i // 64
        b = i % 64
        triple_info_map[triple] = (w, 1 << b)
        
    # Initialize uncovered mask
    uncovered = np.ones(num_words, dtype=np.uint64)
    remainder = num_triples % 64
    if remainder != 0:
        mask_last = (1 << remainder) - 1
        uncovered[-1] &= mask_last
    
    # 2. Precompute block masks
    blocks = list(itertools.combinations(range(v), k))
    N_blocks = len(blocks)
    
    # masks shape: (N_blocks, num_words)
    masks = np.zeros((N_blocks, num_words), dtype=np.uint64)
    
    # Precompute masks
    for i, block in enumerate(blocks):
        for triple in itertools.combinations(block, t):
            w, m = triple_info_map[triple]
            masks[i, w] |= m
            
    # 3. Greedy selection
    selected_blocks = []
    
    while True:
        if time.time() > deadline:
            break
            
        # Check if all covered
        if not np.any(uncovered):
            break
            
        # Calculate gains
        gains = (masks & uncovered).sum(axis=1)
        
        # Find best block
        idx = np.argmax(gains)
        
        if gains[idx] == 0:
            break
            
        selected_blocks.append(list(blocks[idx]))
        
        # Update uncovered
        uncovered &= ~masks[idx]
        
    return selected_blocks
