import itertools
import numpy as np
from math import comb

def construct(v, k, t):
    # 1. Generate all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # 2. Map all t-subsets (triples) to indices
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # 3. Precompute the triple indices for each block
    k_t = comb(k, t)
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_triple_indices[i, j] = triple_to_idx[sub]
            
    # 4. Greedy selection with random tie-breaking
    uncovered_mask = np.ones(n_triples, dtype=bool)
    selected_blocks = []
    
    # We will keep a list of available block indices
    available_blocks = np.arange(n_blocks)
    
    while True:
        if len(available_blocks) == 0:
            break
            
        # Calculate gains for all available blocks
        # uncovered_mask is shape (n_triples,)
        # block_triple_indices[available_blocks] is shape (len(available), k_t)
        # Result of indexing is shape (len(available), k_t) of bools
        gains = np.sum(uncovered_mask[block_triple_indices[available_blocks]], axis=1)
        
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        # Find all blocks with max gain
        max_indices_in_available = np.where(gains == max_gain)[0]
        
        # Randomly pick one from the max gain candidates
        # Using np.random.choice for random selection
        # Note: In a competitive programming/evaluation context, random seed might be fixed or not.
        # To ensure reproducibility or just standard behavior, we might use a global random state.
        # The prompt implies "Implement this with a random...", so we use random.
        chosen_idx_in_available = np.random.choice(max_indices_in_available)
        chosen_block_idx = available_blocks[chosen_idx_in_available]
        
        # Update uncovered mask
        block_triples = block_triple_indices[chosen_block_idx]
        uncovered_mask[block_triples] = False
        
        # Add block to selected
        selected_blocks.append(list(all_blocks[chosen_block_idx]))
        
        # Remove chosen block from available
        # We can't just remove from numpy array easily without copying.
        # Since n_blocks is ~346k, and we select ~76, copying is cheap.
        # Or we can just filter out.
        available_blocks = available_blocks[available_blocks != chosen_block_idx]
        
    return selected_blocks
