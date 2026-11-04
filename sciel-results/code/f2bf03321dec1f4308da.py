import numpy as np
import itertools
import random

D = 11

def construct():
    # Generate all 4-subsets of indices 0..10
    all_subsets = list(itertools.combinations(range(D), 4))
    
    best_blocks = []
    
    # Use multiple random restarts with a greedy approach, 
    # but add a local search step to escape local optima.
    
    for _ in range(2000):
        random.shuffle(all_subsets)
        current_blocks = []
        covered_3s = set()
        
        # Greedy construction
        for block in all_subsets:
            block_3s = list(itertools.combinations(block, 3))
            valid = True
            for t in block_3s:
                if t in covered_3s:
                    valid = False
                    break
            
            if valid:
                current_blocks.append(block)
                for t in block_3s:
                    covered_3s.add(t)
        
        # Local Search: Try to swap a block out for a new one to increase size
        # Since we are maximizing count, we can try to add more if we remove some.
        # But greedy is usually good enough for this small N.
        
        if len(current_blocks) > len(best_blocks):
            best_blocks = current_blocks
            if len(best_blocks) >= 41:
                break
            
    # Construct vectors
    vectors = []
    
    # All 16 sign combinations for 4 dimensions
    all_signs = list(itertools.product([1, -1], repeat=4))
    
    for block in best_blocks:
        for signs in all_signs:
            v = [0] * D
            for idx, sign in zip(block, signs):
                v[idx] = sign
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
