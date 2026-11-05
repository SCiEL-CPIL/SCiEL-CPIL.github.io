import itertools
import numpy as np
from math import comb
import time

def construct(v, k, t):
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # Precompute block_triples and triple_to_blocks
    block_triples = []
    triple_to_blocks = [[] for _ in range(n_triples)]
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        b_triples = []
        for sub in itertools.combinations(block, t):
            t_idx = triple_to_idx[sub]
            b_triples.append(t_idx)
            triple_to_blocks[t_idx].append(i)
            triple_freq[t_idx] += 1
        block_triples.append(b_triples)
    
    # Weighted gain: use inverse frequency as weight (rarer triples are more valuable)
    # Normalize weights so that the sum over a full block's triples gives a meaningful scale
    weights = 1.0 / np.maximum(triple_freq, 1).astype(np.float64)
    
    def greedy_cover(random_seed):
        rng = np.random.RandomState(random_seed)
        
        # Weighted gain for each block: sum of weights of uncovered triples in the block
        gains = np.zeros(n_blocks, dtype=np.float64)
        for i in range(n_blocks):
            gains[i] = np.sum(weights[block_triples[i]])
        
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_mask = np.zeros(n_blocks, dtype=bool)
        
        selected_blocks = []
        remaining = n_triples
        
        while remaining > 0:
            if time.time() > deadline:
                break
            
            # Mask out selected blocks
            gains[selected_mask] = -1.0
            
            max_gain = np.max(gains)
            if max_gain <= 0:
                break
                
            # Find candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[rng.randint(len(candidates))]
            
            # Select block
            selected_mask[best_idx] = True
            selected_blocks.append(list(all_blocks[best_idx]))
            
            # Update gains for affected blocks
            newly_covered = []
            for t_idx in block_triples[best_idx]:
                if uncovered_mask[t_idx]:
                    uncovered_mask[t_idx] = False
                    newly_covered.append(t_idx)
                    remaining -= 1
            
            if newly_covered:
                for t_idx in newly_covered:
                    affected = triple_to_blocks[t_idx]
                    if affected:
                        gains[affected] -= weights[t_idx]
                        
        return selected_blocks

    best_blocks = None
    best_len = float('inf')
    
    n_restarts = 100  # Upper bound; time-limited
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        
        blocks = greedy_cover(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    # If no valid result yet (shouldn't happen), do one more with seed 0
    if best_blocks is None:
        best_blocks = greedy_cover(0)
    
    return best_blocks
