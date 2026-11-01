import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 150.0
    
    # 1. Generate all t-subsets and k-blocks
    t_subsets = list(itertools.combinations(range(v), t))
    N_t = len(t_subsets)
    t_subset_to_id = {s: i for i, s in enumerate(t_subsets)}
    
    k_blocks = list(itertools.combinations(range(v), k))
    N_k = len(k_blocks)
    block_to_index = {b: i for i, b in enumerate(k_blocks)}
    
    # 2. Map each k-block to t-subset IDs
    block_to_t_ids = []
    for b in k_blocks:
        t_ids = [t_subset_to_id[ts] for ts in itertools.combinations(b, t)]
        block_to_t_ids.append(t_ids)
        
    # Convert to numpy arrays for fast access
    block_to_t_ids_np = [np.array(ids, dtype=np.int32) for ids in block_to_t_ids]
    
    # 3. Build incidence matrix M (N_k x N_t)
    M = np.zeros((N_k, N_t), dtype=np.int8)
    for j, ids in enumerate(block_to_t_ids):
        M[j, ids] = 1
        
    best_blocks = None
    best_score = float('inf')
    
    seed = 0
    while time.time() < deadline - 10:
        seed += 1
        random.seed(seed)
        np.random.seed(seed)
        
        uncovered = np.ones(N_t, dtype=np.int8)
        blocks = []
        
        valid_covering = False
        while True:
            if time.time() > deadline - 5:
                break # Abort this run if close to deadline
                
            # Calculate gains
            gains = M @ uncovered
            gains = gains.astype(np.int32)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                valid_covering = True
                break # All covered
                
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) == 0:
                break
            idx = random.choice(candidates)
            
            blocks.append(k_blocks[idx])
            
            # Update uncovered
            uncovered[M[idx] == 1] = 0
            
        if not valid_covering:
            continue
            
        # Local Search: Remove redundant blocks
        # Compute coverage counts
        cov = np.zeros(N_t, dtype=np.int32)
        for b in blocks:
            idx = block_to_index[b]
            ids = block_to_t_ids_np[idx]
            cov[ids] += 1
            
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b in blocks:
                idx = block_to_index[b]
                ids = block_to_t_ids_np[idx]
                
                is_redundant = True
                # Check if any t-subset in this block is covered ONLY by this block
                for tid in ids:
                    if cov[tid] == 1:
                        is_redundant = False
                        break
                        
                if is_redundant:
                    # Remove block
                    for tid in ids:
                        cov[tid] -= 1
                    changed = True
                else:
                    new_blocks.append(b)
                    
            blocks = new_blocks
            
        if len(blocks) < best_score:
            best_score = len(blocks)
            best_blocks = blocks[:]
            
    # Convert tuples to lists
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        return []
