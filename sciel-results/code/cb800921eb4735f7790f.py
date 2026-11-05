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
    
    # block_triple_indices[i] is a list of triple indices in block i
    # triple_to_blocks[j] is a list of block indices containing triple j
    block_triple_indices = [None] * n_blocks
    triple_to_blocks = [[] for _ in range(n_triples)]
    
    for i, block in enumerate(all_blocks):
        subs = list(itertools.combinations(block, t))
        idxs = [triple_to_idx[sub] for sub in subs]
        block_triple_indices[i] = idxs
        for idx in idxs:
            triple_to_blocks[idx].append(i)

    def greedy_cover(random_seed):
        rng = np.random.RandomState(random_seed)
        # gain[i] is the number of uncovered triples in block i
        gain = np.full(n_blocks, k_t, dtype=np.int32)
        covered = np.zeros(n_triples, dtype=bool)
        selected_blocks = []
        remaining = n_triples
        
        while remaining > 0:
            # Find max gain
            max_gain = np.max(gain)
            if max_gain == 0:
                break
                
            # Find candidates with max gain
            # Using np.where on a large array is fast
            candidates = np.where(gain == max_gain)[0]
            
            # Randomly pick one
            best_idx = candidates[rng.randint(len(candidates))]
            
            selected_blocks.append(list(all_blocks[best_idx]))
            
            # Update gains for blocks sharing triples with best_idx
            for triple_idx in block_triple_indices[best_idx]:
                if not covered[triple_idx]:
                    covered[triple_idx] = True
                    remaining -= 1
                    # Decrement gain for all blocks containing this triple
                    for b_idx in triple_to_blocks[triple_idx]:
                        gain[b_idx] -= 1
                        
        return selected_blocks

    # Run multiple times
    best_blocks = None
    best_len = float('inf')
    
    # Since the algorithm is faster, we can do more restarts
    # 20 restarts in the old code might have taken a long time.
    # Let's try 50 restarts to see if we can find a better solution.
    n_restarts = 50
    
    for i in range(n_restarts):
        blocks = greedy_cover(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks
