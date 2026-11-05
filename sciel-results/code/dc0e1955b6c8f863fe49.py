import itertools
import numpy as np
from math import comb
import scipy.sparse as sp

def construct(v, k, t):
    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # Store block indices for fast access
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_triple_indices[i, j] = triple_to_idx[sub]
            
    # Build sparse matrix
    row_indices = np.repeat(np.arange(n_blocks), k_t)
    col_indices = block_triple_indices.flatten()
    data = np.ones(len(col_indices), dtype=np.int8)
    M = sp.coo_matrix((data, (row_indices, col_indices)), shape=(n_blocks, n_triples)).tocsr()
    
    def greedy_cover(random_seed):
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        remaining = n_triples
        
        while remaining > 0:
            # Calculate gains
            gains = M.dot(uncovered_mask)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[rng.randint(len(candidates))]
            
            # Update state
            start = M.indptr[best_idx]
            end = M.indptr[best_idx + 1]
            block_triples = M.indices[start:end]
            
            uncovered_mask[block_triples] = False
            remaining -= max_gain
            selected_blocks.append(list(all_blocks[best_idx]))
            
        return selected_blocks

    best_blocks = None
    best_len = float('inf')
    
    n_restarts = 50
    
    for i in range(n_restarts):
        blocks = greedy_cover(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks
