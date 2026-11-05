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
    
    # block_triple_indices[i, j] is the index of the j-th triple in block i
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_triple_indices[i, j] = triple_to_idx[sub]
            
    def greedy_cover(random_seed):
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        remaining = n_triples
        
        # Track triples covered by selected blocks for tie-breaking (redundancy penalty)
        selected_triples_mask = np.zeros(n_triples, dtype=bool)
        
        while remaining > 0:
            # Calculate gains: number of uncovered triples in each block
            # Fancy indexing creates a (n_blocks, k_t) boolean array
            gains = np.sum(uncovered_mask[block_triple_indices], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            if len(candidates) == 1:
                best_idx = candidates[0]
            else:
                # Tie-breaking: Penalize redundancy with already selected blocks.
                # We prefer blocks that have LESS overlap with previously selected blocks' triples.
                # This preserves the "diversity" of the covering.
                
                # Calculate overlap of each candidate with selected_triples_mask
                # candidate_triples shape: (n_candidates, k_t)
                candidate_triples = block_triple_indices[candidates]
                
                # overlap_counts[i] = number of triples in candidate i that are already in selected_triples_mask
                # selected_triples_mask is boolean (n_triples,)
                # candidate_triples is int (n_candidates, k_t)
                # We need to check how many of the triples in candidate_triples are True in selected_triples_mask
                overlaps = np.sum(selected_triples_mask[candidate_triples], axis=1)
                
                # Choose the candidate with the minimum overlap.
                # If there are still ties among min-overlap candidates, pick randomly.
                min_overlap = np.min(overlaps)
                min_overlap_candidates = candidates[overlaps == min_overlap]
                
                best_idx = min_overlap_candidates[rng.randint(len(min_overlap_candidates))]
            
            # Update state
            block_idx = block_triple_indices[best_idx]
            uncovered_mask[block_idx] = False
            selected_triples_mask[block_idx] = True
            remaining -= max_gain
            selected_blocks.append(list(all_blocks[best_idx]))
            
        return selected_blocks

    # Run multiple times with different seeds
    best_blocks = None
    best_len = float('inf')
    
    # 30 restarts to allow more exploration
    n_restarts = 30
    
    for i in range(n_restarts):
        blocks = greedy_cover(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks
