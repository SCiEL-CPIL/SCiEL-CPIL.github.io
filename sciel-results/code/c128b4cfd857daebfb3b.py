import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    D = 11
    
    # Type A: Vectors with one non-zero entry +/- 2
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
        
    vectors_A = np.array(vectors_A, dtype=np.int64)
    
    # Type D: Vectors with four non-zero entries +/- 1
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    
    # Generate all Type D vectors
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
            
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)  # 5280
    
    # Compute dot products between all Type D vectors
    dots = all_vectors_D @ all_vectors_D.T  # (5280, 5280)
    
    # Conflict if dot >= 3 (since we need dot <= 2 for ||u-v||^2 >= 4)
    conflict = dots >= 3
    np.fill_diagonal(conflict, True)
    
    # Degrees
    degrees = np.sum(conflict, axis=1)
    
    # Greedy: sort by degree ascending, pick in order if compatible
    order = np.argsort(degrees)
    
    best_set = []
    best_size = 0
    
    # First pass: min-degree order
    selected_mask = np.zeros(N_all_D, dtype=bool)
    selected = []
    for idx in order:
        if selected_mask[idx]:
            continue
        if not np.any(conflict[idx] & selected_mask):
            selected.append(idx)
            selected_mask[idx] = True
    
    best_size = len(selected)
    best_set = selected
    
    # Try random orderings for better results
    num_random_tries = 50
    for r in range(num_random_tries):
        if time.time() > deadline - 10:
            break
        random_order = np.random.permutation(N_all_D)
        selected_mask = np.zeros(N_all_D, dtype=bool)
        selected = []
        for idx in random_order:
            if selected_mask[idx]:
                continue
            if not np.any(conflict[idx] & selected_mask):
                selected.append(idx)
                selected_mask[idx] = True
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
    
    # Extract selected Type D vectors
    selected_vectors_D = all_vectors_D[best_set]
    
    # Combine Type A and Type D
    final_vectors = np.vstack([vectors_A, selected_vectors_D])
    
    return final_vectors
