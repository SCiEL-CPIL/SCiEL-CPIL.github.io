import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    # 1. Generate all t-subsets and map to indices
    all_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # 2. Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # 3. Build incidence structures
    # block_to_subsets[b] = list of subset indices in block b
    block_to_subsets = [[] for _ in range(num_blocks)]
    # subset_to_blocks[u] = list of block indices containing subset u
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for subset in itertools.combinations(block, t):
            u_idx = subset_to_idx[subset]
            block_to_subsets[b_idx].append(u_idx)
            subset_to_blocks[u_idx].append(b_idx)
    
    # 4. Initialize state
    uncovered = set(range(num_subsets))
    gains = np.zeros(num_blocks, dtype=np.int32)
    
    # Initial gain calculation
    for u in uncovered:
        for b in subset_to_blocks[u]:
            gains[b] += 1
            
    blocks_found = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Find block with max gain
        b_star = np.argmax(gains)
        
        # Add block to solution
        blocks_found.append(list(all_blocks[b_star]))
        
        # Determine which subsets are covered by this block and were uncovered
        covered_by_star = []
        for u in block_to_subsets[b_star]:
            if u in uncovered:
                covered_by_star.append(u)
                
        # Remove them from uncovered
        uncovered.difference_update(covered_by_star)
        
        # If all covered, stop
        if not uncovered:
            break
            
        # Update gains
        # For each subset covered, decrement gain for all blocks containing it
        for u in covered_by_star:
            for b in subset_to_blocks[u]:
                gains[b] -= 1
                
    return blocks_found
