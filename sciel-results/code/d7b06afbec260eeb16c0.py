"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate candidate vectors with norm sqrt(3) (3 non-zero entries, +/- 1)
    candidates = []
    for support in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for i, s in zip(support, signs):
                v[i] = s
            candidates.append(v)
    
    n_cands = len(candidates)
    
    # Precompute dot products? 1320 * 1320 is ~1.7M ops. Fast.
    # But we only need to check compatibility.
    # Condition: v . w <= 1
    
    # Greedy initialization
    # Sort by "popularity" or just random shuffle
    indices = list(range(n_cands))
    random.shuffle(indices)
    
    selected_indices = []
    selected_vectors = []
    
    for idx in indices:
        v = candidates[idx]
        is_compatible = True
        for w in selected_vectors:
            if np.dot(v, w) > 1:
                is_compatible = False
                break
        if is_compatible:
            selected_indices.append(idx)
            selected_vectors.append(v)
            
    best_count = len(selected_indices)
    best_set = list(selected_indices)
    
    # Simulated Annealing / Local Search
    # Try to swap out a vector to allow more in?
    # Or just try to add more?
    # A simple improvement: iterate through unselected, try to add.
    # If stuck, try to remove a random selected vector and re-greedy?
    
    # Let's do a simple hill climbing:
    # 1. Try to add any unselected vector.
    # 2. If no vector can be added, try a swap: remove one, add two?
    
    # For speed, we will do a few passes of "try to add".
    # Then, if time permits, do random restarts or swaps.
    
    unselected = set(indices) - set(selected_indices)
    
    # Pass 1: Greedy add
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in list(unselected):
            v = candidates[idx]
            is_compatible = True
            for w in selected_vectors:
                if np.dot(v, w) > 1:
                    is_compatible = False
                    break
            if is_compatible:
                selected_indices.append(idx)
                selected_vectors.append(v)
                unselected.remove(idx)
                changed = True
                if len(selected_indices) > best_count:
                    best_count = len(selected_indices)
                    best_set = list(selected_indices)
    
    # If we have time, try random restarts with different shuffles
    # Or try a different norm?
    # Let's stick to this norm for now.
    
    # To potentially beat 220 significantly, we need > 220.
    # Greedy on 1320 nodes usually gets a decent fraction.
    
    # Let's try one more strategy: 
    # If the set is small, try norm sqrt(2) vectors (the seed) as a fallback.
    # Seed count is 220.
    
    seed_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                seed_vectors.append(v)
    
    if best_count < len(seed_vectors):
        return np.array(seed_vectors, dtype=np.int64)
        
    result = np.array([candidates[i] for i in best_set], dtype=np.int64)
    return result
# EVOLVE-BLOCK-END
