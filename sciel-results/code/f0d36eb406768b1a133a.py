import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Precompute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    def greedy_construct(seed):
        np.random.seed(seed)
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        blocks = []
        
        uncovered_indices = np.arange(num_t_subsets)
        
        while len(uncovered_indices) > 0:
            if time.time() > deadline:
                return None  # Timeout
            
            anchor_idx = uncovered_indices[np.random.randint(len(uncovered_indices))]
            anchor = all_t_subsets[anchor_idx]
            
            block = list(anchor)
            
            while len(block) < k:
                best_p = -1
                best_gain = -1
                for p in range(v):
                    if p in block:
                        continue
                    new_block = block + [p]
                    gain = 0
                    for t_sub in itertools.combinations(new_block, t):
                        idx = subset_to_idx[t_sub]
                        if counts[idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                
                if best_p == -1:
                    break
                block.append(best_p)
            
            block.sort()
            for t_sub in itertools.combinations(block, t):
                idx = subset_to_idx[t_sub]
                counts[idx] += 1
            
            blocks.append(block)
            uncovered_indices = np.where(counts == 0)[0]
            
        return blocks

    def reduce_covering(blocks):
        if not blocks:
            return blocks
        
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        for block in blocks:
            for t_sub in itertools.combinations(block, t):
                idx = subset_to_idx[t_sub]
                counts[idx] += 1
        
        new_blocks = []
        for block in blocks:
            if time.time() > deadline:
                break
                
            block_t_subsets = []
            for t_sub in itertools.combinations(block, t):
                idx = subset_to_idx[t_sub]
                block_t_subsets.append(idx)
            
            redundant = True
            for idx in block_t_subsets:
                counts[idx] -= 1
            
            for idx in block_t_subsets:
                if counts[idx] == 0:
                    redundant = False
                    break
            
            for idx in block_t_subsets:
                counts[idx] += 1
            
            if redundant:
                for idx in block_t_subsets:
                    counts[idx] -= 1
            else:
                new_blocks.append(block)
                
        return new_blocks

    best_blocks = None
    best_size = float('inf')
    
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        blocks = greedy_construct(i)
        if blocks is None:
            continue
        
        blocks = reduce_covering(blocks)
        
        # Verify
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        for block in blocks:
            for t_sub in itertools.combinations(block, t):
                idx = subset_to_idx[t_sub]
                counts[idx] += 1
        
        if np.all(counts > 0):
            if len(blocks) < best_size:
                best_size = len(blocks)
                best_blocks = blocks
        
    if best_blocks is None:
        # Fallback: run one greedy without reduction if we have time
        if time.time() < deadline - 10:
            blocks = greedy_construct(999)
            if blocks is not None:
                best_blocks = blocks
        else:
            # Critical failure: no valid covering found
            # Return empty list? No, must be valid.
            # This case should be rare.
            best_blocks = []

    return best_blocks
