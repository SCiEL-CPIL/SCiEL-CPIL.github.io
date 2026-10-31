import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subsets_set = {tuple(s): i for i, s in enumerate(t_subsets)}
    num_tsubs = len(t_subsets)
    
    # Generate all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(k_subsets)
    
    # Precompute for each block, which t-subsets it covers
    # block_to_tsubs[i] is a list of indices of t-subsets covered by block i
    block_to_tsubs = []
    for block in k_subsets:
        covered_tsubs = []
        for tsub in itertools.combinations(block, t):
            idx = t_subsets_set[tsub]
            covered_tsubs.append(idx)
        block_to_tsubs.append(covered_tsubs)
        
    # Precompute for each t-subset, which blocks cover it
    # tsub_to_blocks[j] is a list of indices of blocks covering t-subset j
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for i, tsubs in enumerate(block_to_tsubs):
        for s in tsubs:
            tsub_to_blocks[s].append(i)
            
    # Convert tsub_to_blocks lists to numpy arrays for faster indexing
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # Initial gains: each block covers C(k, t) t-subsets
    k_choose_t = len(block_to_tsubs[0])
    gains = np.full(num_blocks, k_choose_t, dtype=np.int32)
    
    # Track uncovered t-subsets
    uncovered = np.ones(num_tsubs, dtype=bool)
    
    selected_block_indices = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
            
        # Find the block with the maximum gain
        b = np.argmax(gains)
        
        if gains[b] == 0:
            # Should not happen if uncovered.any() is true, but safety check
            break
            
        selected_block_indices.append(b)
        
        # Update gains for all blocks that share any t-subset with b
        # We only decrement gains for t-subsets that were previously uncovered
        for s in block_to_tsubs[b]:
            if uncovered[s]:
                uncovered[s] = False
                # Decrement gain for all blocks that cover s
                # Using numpy for vectorized subtraction
                gains[tsub_to_blocks_np[s]] -= 1

    # Convert selected block indices back to the actual subsets
    result = [list(k_subsets[i]) for i in selected_block_indices]
    
    return result
