import time
import numpy as np
from itertools import combinations

D = 11

def construct():
    deadline = time.time() + 140  # 140s safety margin

    # Type A vectors: (±2, 0, ..., 0)
    type_a = []
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            type_a.append(v)
    
    # Type B vectors: four ±1 entries
    # We need to find the maximum independent set among these
    # Incompatibility: dot product > 2
    # For Type B, dot product > 2 implies dot product = 3 or 4.
    # dot = 4: identical (we generate unique ones)
    # dot = 3: share exactly 3 coordinates with same signs.
    
    # We will represent each Type B vector by its set of 3-grams.
    # A 3-gram is a (index1, index2, index3, sign1, sign2, sign3) tuple.
    # We can map these to integers for fast set operations.
    
    # Precompute mapping from 3-gram to ID
    # Indices: 0..10
    # Signs: 0 for -1, 1 for 1 (or just use the value)
    # Let's just use a tuple (i, j, k, s_i, s_j, s_k) with i<j<k.
    
    # Generate all Type B vectors
    type_b_vectors = []
    type_b_grams = []
    
    for combo in combinations(range(D), 4):
        for signs in range(16):
            v = [0] * D
            gram_set = set()
            # Assign signs
            vals = []
            for k in range(4):
                val = 1 if (signs >> k) & 1 else -1
                vals.append(val)
                v[combo[k]] = val
            
            # Generate 3-grams
            # Choose 3 indices from the 4
            for c3 in combinations(range(4), 3):
                idxs = tuple(combo[x] for x in c3)
                sigs = tuple(vals[x] for x in c3)
                gram_set.add((idxs, sigs))
            
            type_b_vectors.append(v)
            type_b_grams.append(gram_set)
            
    n_b = len(type_b_vectors)
    
    # Local search to find max independent set
    # State: set of selected indices
    # We use a greedy initialization and then local search.
    
    best_set = set()
    best_grams = set()
    
    # Greedy initialization
    indices = list(range(n_b))
    np.random.shuffle(indices)
    
    for idx in indices:
        grams = type_b_grams[idx]
        if not (grams & best_grams):
            best_set.add(idx)
            best_grams.update(grams)
            
    # Local search: 1-swap
    # Try to replace a selected vector with an unselected one
    # If we remove v_in, we free up its grams.
    # If we add v_out, we use its grams.
    # Condition: v_out is compatible with (best_set - {v_in})
    # i.e., grams(v_out) is disjoint from (best_grams - grams(v_in))
    
    # To speed up, we can just iterate.
    # 5280 vectors.
    # We run for a fixed number of iterations or time.
    
    selected_list = list(best_set)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Shuffle to avoid local minima
        np.random.shuffle(selected_list)
        
        for v_in in selected_list:
            grams_in = type_b_grams[v_in]
            # Tentatively remove
            temp_grams = best_grams - grams_in
            
            # Find a candidate v_out not in best_set that is compatible
            # We can scan all not-in-set vectors
            # To make it faster, we can just check a few random ones or all.
            # 5280 is small.
            
            # Optimization: only check v_out if it conflicts with v_in?
            # Not necessarily, but likely.
            
            for v_out in range(n_b):
                if v_out in best_set:
                    continue
                grams_out = type_b_grams[v_out]
                if not (grams_out & temp_grams):
                    # Valid swap
                    best_set.remove(v_in)
                    best_set.add(v_out)
                    best_grams = temp_grams | grams_out
                    improved = True
                    break # Restart scan for this v_in? Or just continue?
                    # Usually better to restart the loop to update selected_list
                    break
                    
            if improved:
                selected_list = list(best_set)
                break # Restart outer loop
                
    # Combine Type A and selected Type B
    final_vectors = np.array(type_a, dtype=np.int64)
    if best_set:
        selected_b = np.array([type_b_vectors[i] for i in best_set], dtype=np.int64)
        final_vectors = np.vstack([final_vectors, selected_b])
        
    return final_vectors
