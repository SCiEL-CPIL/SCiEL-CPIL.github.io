import itertools
import numpy as np
from math import comb

def construct(v, k, t):
    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # Store block indices for fast access
    # block_triple_indices[i, j] is the index of the j-th triple in block i
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    
    # Prepare triple_to_blocks as list of lists first
    triple_to_blocks_list = [[] for _ in range(n_triples)]
    
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            idx = triple_to_idx[sub]
            block_triple_indices[i, j] = idx
            triple_to_blocks_list[idx].append(i)
            
    # Convert to numpy arrays for vectorized updates
    triple_to_blocks = [np.array(lst, dtype=np.int32) for lst in triple_to_blocks_list]
    
    def greedy_cover(random_seed):
        rng = np.random.RandomState(random_seed)
        # Initially, all triples are uncovered, so gain of each block is k_t
        block_gains = np.full(n_blocks, k_t, dtype=np.int32)
        selected_blocks = []
        remaining = n_triples
        
        while remaining > 0:
            max_gain = np.max(block_gains)
            if max_gain == 0:
                break
                
            candidates = np.where(block_gains == max_gain)[0]
            best_idx = candidates[rng.randint(len(candidates))]
            
            selected_blocks.append(list(all_blocks[best_idx]))
            remaining -= max_gain
            
            # Update gains for all blocks
            # Get the triple indices covered by the selected block
            triples_covered = block_triple_indices[best_idx]
            for t_idx in triples_covered:
                # Subtract 1 from gain of all blocks containing this triple
                block_gains[triple_to_blocks[t_idx]] -= 1
                
        return selected_blocks

    # Run multiple restarts
    best_blocks = None
    best_len = float('inf')
    
    # Increase restarts since it's faster now
    # 120s limit. Precomp ~2s.
    # Each restart ~76 iterations * (max + update).
    # Update is 35 vector ops of size ~6000. Very fast.
    # Let's try 100 restarts.
    n_restarts = 100
    
    for i in range(n_restarts):
        blocks = greedy_cover(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks
