import numpy as np
import itertools
from collections import deque

def construct(v, k, t):
    # 1. Generate all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # 2. Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(k_blocks)
    
    # 3. Precompute block_t_list: for each block, list of t-subset indices
    block_t_list = []
    for block in k_blocks:
        covered = [t_index[ts] for ts in itertools.combinations(block, t)]
        block_t_list.append(covered)
        
    # 4. Compute initial coverage counts
    coverage = np.zeros(n_t, dtype=np.int32)
    for covered in block_t_list:
        for idx in covered:
            coverage[idx] += 1
            
    # 5. Build blocks_covering: for each t-subset, list of block indices
    # We need this to find the "other" block when count drops to 1
    blocks_covering = [[] for _ in range(n_t)]
    for i, covered in enumerate(block_t_list):
        for idx in covered:
            blocks_covering[idx].append(i)
            
    # 6. Compute essential_count for each block
    # essential_count[i] = number of t-subsets in block i that have coverage == 1
    essential_count = np.zeros(n_k, dtype=np.int32)
    for i, covered in enumerate(block_t_list):
        for idx in covered:
            if coverage[idx] == 1:
                essential_count[i] += 1
                
    # 7. Initialize queue with removable blocks (essential_count == 0)
    active = np.ones(n_k, dtype=bool)
    queue = deque()
    for i in range(n_k):
        if essential_count[i] == 0:
            queue.append(i)
            
    # 8. Process queue
    while queue:
        b = queue.popleft()
        if not active[b]:
            continue
        if essential_count[b] > 0:
            continue
            
        # Remove block b
        active[b] = False
        
        for idx in block_t_list[b]:
            old_count = coverage[idx]
            coverage[idx] -= 1
            new_count = coverage[idx]
            
            # If count drops to 1, the other active block covering idx becomes essential for idx
            if new_count == 1:
                # Find the other active block
                # We know old_count was 2 (since new_count is 1 and we decremented by 1)
                # So there is exactly one other active block
                for b_prime in blocks_covering[idx]:
                    if b_prime != b and active[b_prime]:
                        # b_prime is the other one
                        old_essential = essential_count[b_prime]
                        essential_count[b_prime] = old_essential + 1
                        # If it was 0, it might be in the queue.
                        # If it was > 0, it stays out.
                        # We don't need to add it to queue if it becomes > 0.
                        # If it was 0 and becomes 1, it is no longer removable.
                        # It might be in the queue, but we check `essential_count > 0` when popping.
                        break
                        
    # 9. Collect results
    result = [list(k_blocks[i]) for i in range(n_k) if active[i]]
    
    return result
