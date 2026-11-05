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
    
    # Store block indices for fast access
    # block_triple_indices[i, j] is the index of the j-th triple in block i
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    
    # Calculate frequency of each triple to determine weights
    # Triples that appear in fewer blocks are "rarer" and should be prioritized
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            idx = triple_to_idx[sub]
            block_triple_indices[i, j] = idx
            triple_freq[idx] += 1
            
    # Weights are inversely proportional to frequency
    # This implements a weighted greedy heuristic
    triple_freq = np.maximum(triple_freq, 1)
    weights = 1.0 / triple_freq.astype(np.float64)
    
    def greedy_cover(random_seed):
        # Use a local RandomState to avoid global side effects
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        remaining = n_triples
        
        while remaining > 0:
            # Calculate weighted gains
            # Only uncovered triples contribute to the gain
            active_weights = np.where(uncovered_mask, weights, 0.0)
            
            # Sum weights for each block
            gains = np.sum(active_weights[block_triple_indices], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one candidate to break ties randomly
            best_idx = candidates[rng.randint(len(candidates))]
            
            # Update state
            block_triples = block_triple_indices[best_idx]
            
            # Count newly covered triples to update remaining count
            newly_covered = np.sum(uncovered_mask[block_triples])
            remaining -= newly_covered
            
            # Mark as covered
            uncovered_mask[block_triples] = False
            
            selected_blocks.append(list(all_blocks[best_idx]))
            
        return selected_blocks

    # Run multiple times with different seeds to find a better covering
    best_blocks = None
    best_len = float('inf')
    
    # Slightly more restarts to compensate for potential local optima in weighted heuristic
    n_restarts = 25
    
    for i in range(n_restarts):
        blocks = greedy_cover(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks
