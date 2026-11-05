import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    # 1. Generate all t-subsets and map to indices
    all_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # 2. Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # 3. Build incidence structures as arrays for speed
    # block_to_subsets[b] = array of subset indices in block b
    block_to_subsets = []
    # subset_to_blocks[u] = array of block indices containing subset u
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        subs = []
        for subset in itertools.combinations(block, t):
            u_idx = subset_to_idx[subset]
            subs.append(u_idx)
            subset_to_blocks[u_idx].append(b_idx)
        block_to_subsets.append(np.array(subs, dtype=np.int32))
    
    # Convert subset_to_blocks to arrays
    subset_to_blocks_arr = [np.array(lst, dtype=np.int32) for lst in subset_to_blocks]
    
    # 4. Initialize state with NumPy arrays
    uncovered = np.ones(num_subsets, dtype=bool)
    gains = np.zeros(num_blocks, dtype=np.int32)
    
    # Initial gain calculation: for each subset, add 1 to all blocks containing it
    for u in range(num_subsets):
        if uncovered[u]:
            gains[subset_to_blocks_arr[u]] += 1
            
    blocks_found = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
            
        # Find block with max gain
        b_star = int(np.argmax(gains))
        
        # Add block to solution
        blocks_found.append(list(all_blocks[b_star]))
        
        # Determine which subsets are covered by this block and were uncovered
        block_subs = block_to_subsets[b_star]
        newly_covered = block_subs[uncovered[block_subs]]
        
        # Remove them from uncovered
        if len(newly_covered) > 0:
            uncovered[newly_covered] = False
            
        # If all covered, stop
        if not uncovered.any():
            break
            
        # Update gains: for each newly covered subset, decrement gain for all blocks containing it
        if len(newly_covered) > 0:
            for u in newly_covered:
                gains[subset_to_blocks_arr[u]] -= 1
                
    # If we have time, try a simple local improvement: swap blocks to reduce count
    # This is a refinement pass - try to find redundant blocks
    remaining_time = deadline - time.time()
    if remaining_time > 10 and len(blocks_found) > 1:
        # Try removing each block and checking if coverage is still complete
        # Only do this if we have significant time left
        if remaining_time > 30:
            n_blocks = len(blocks_found)
            # Build a coverage matrix: which t-subsets are covered by which blocks
            # For efficiency, only try this if num_blocks is manageable
            for remove_idx in range(n_blocks - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                # Check if removing block remove_idx still covers all subsets
                # Count how many blocks cover each subset
                cover_count = np.zeros(num_subsets, dtype=np.int32)
                for bi, block in enumerate(blocks_found):
                    if bi == remove_idx:
                        continue
                    for subset in itertools.combinations(block, t):
                        u_idx = subset_to_idx[subset]
                        cover_count[u_idx] += 1
                if cover_count.min() >= 1:
                    blocks_found.pop(remove_idx)
                    break
                    
    return blocks_found
