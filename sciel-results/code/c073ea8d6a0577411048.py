"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 55  # safety margin before 60s limit
    
    # Pre-compute all 4-subsets of [1,16]
    all_quads = list(combinations(range(1, 17), 4))
    n_quads = len(all_quads)  # 1820
    
    # Represent each set as a 16-element binary numpy array
    set_matrix = np.zeros((n_quads, 16), dtype=np.int8)
    for i, quad in enumerate(all_quads):
        for elem in quad:
            set_matrix[i, elem - 1] = 1
    
    # Precompute pairwise intersection bitmasks
    # inter_mask[i, j] = bitmask of elements in all_quads[i] ∩ all_quads[j]
    inter_mask = np.zeros((n_quads, n_quads), dtype=np.int32)
    for k in range(16):
        idxs = np.where(set_matrix[:, k] == 1)[0]
        for ii in range(len(idxs)):
            for jj in range(ii + 1, len(idxs)):
                inter_mask[idxs[ii], idxs[jj]] |= (1 << k)
                inter_mask[idxs[jj], idxs[ii]] |= (1 << k)
    
    best_family = []
    best_size = 0
    
    # Multiple random restarts with greedy
    num_restarts = 200
    for trial in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(trial * 104729 + 7)
        order = list(range(n_quads))
        random.shuffle(order)
        
        current_idx = []  # indices of selected sets
        current_size = 0
        
        for idx in order:
            if time.time() > deadline:
                break
            
            # Check if adding idx creates a sunflower with any two existing sets
            valid = True
            for a_pos in range(current_size):
                if not valid:
                    break
                a = current_idx[a_pos]
                sa = inter_mask[idx, a]
                for b_pos in range(a_pos + 1, current_size):
                    b = current_idx[b_pos]
                    sb = inter_mask[idx, b]
                    ab = inter_mask[a, b]
                    if sa == sb == ab:
                        valid = False
                        break
            
            if valid:
                current_idx.append(idx)
                current_size += 1
        
        if current_size > best_size:
            best_size = current_size
            best_family = current_idx[:]
        
        # Early termination if we hit the max
        if best_size >= 250:
            break
    
    # Convert back to sorted lists
    result = [sorted(all_quads[i]) for i in best_family]
    return result
# EVOLVE-BLOCK-END
