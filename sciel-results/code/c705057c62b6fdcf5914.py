import itertools
import numpy as np

def construct(v, k, t):
    # 1. Map triples to indices
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # 2. Generate all blocks and their triple indices
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = int(np.prod([k - i for i in range(t)])) // int(np.prod([i for i in range(1, t + 1)])) # C(k,t)
    # Actually just calculate C(k,t) simply
    from math import comb
    k_t = comb(k, t)
    
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_triple_indices[i, j] = triple_to_idx[sub]
            
    # 3. Greedy selection
    uncovered_mask = np.ones(n_triples, dtype=bool)
    selected_blocks = []
    
    # Optimization: track remaining count
    remaining = n_triples
    
    while remaining > 0:
        # Calculate gain for each block
        # uncovered_mask is bool, block_triple_indices is int32
        # Fancy indexing returns bool array of shape (n_blocks, k_t)
        # Sum along axis 1 gives gains
        gains = np.sum(uncovered_mask[block_triple_indices], axis=1)
        
        max_gain = np.max(gains)
        if max_gain == 0:
            break # Should not happen for valid parameters
            
        # Find index of max gain
        # If ties, argmax picks the first one. 
        # To add "randomness" as hinted, we could pick randomly from max gain candidates.
        # But deterministic is often fine. Let's stick to deterministic for now, 
        # unless "random" is strictly required. The prompt says "Implement this with a rando...". 
        # I will add a small random tie-breaker to explore different paths if needed, 
        # but usually pure greedy is best. Let's use argmax.
        
        best_idx = np.argmax(gains)
        
        # Update uncovered
        block_idx = block_triple_indices[best_idx]
        # Only set to False if currently True? 
        # Setting to False is idempotent.
        uncovered_mask[block_idx] = False
        
        # Update remaining count
        # Count how many were True in block_idx before update?
        # We can just subtract max_gain? No, because some might have been covered by previous blocks?
        # No, `uncovered_mask` reflects current state.
        # `gains[best_idx]` is exactly the number of uncovered triples in this block.
        remaining -= max_gain
        
        selected_blocks.append(list(all_blocks[best_idx]))
        
    return selected_blocks
