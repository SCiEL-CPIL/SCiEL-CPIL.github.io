import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    
    # 1. Generate all weight 4 vectors
    vectors = []
    vector_indices = [] # To map back to original
    for comb in combinations(range(d), 4):
        for signs in product([-1, 1], repeat=4):
            p = np.zeros(d, dtype=np.int64)
            for idx, val in zip(comb, signs):
                p[idx] = val
            vectors.append(p)
            vector_indices.append((comb, signs))
            
    V = np.array(vectors)
    N = len(V)
    
    # 2. Pre-compute 3-tuple IDs
    # Map (tuple of 3 indices, tuple of 3 signs) to ID
    # There are 1320 such tuples.
    id_map = {}
    current_id = 0
    # Generate all possible 3-tuples
    # Indices: combinations(range(11), 3)
    # Signs: product([-1, 1], repeat=3)
    for c3 in combinations(range(d), 3):
        for s3 in product([-1, 1], repeat=3):
            id_map[(c3, s3)] = current_id
            current_id += 1
    max_id = current_id
            
    # For each vector, find its 4 associated 3-tuple IDs
    vector_tuples = []
    for comb, signs in vector_indices:
        tups = []
        # comb is 4 indices, signs is 4 signs
        # 4 ways to pick 3
        for k in range(4):
            # Remove index k
            c3 = tuple(comb[i] for i in range(4) if i != k)
            s3 = tuple(signs[i] for i in range(4) if i != k)
            # Get ID
            tid = id_map[(c3, s3)]
            tups.append(tid)
        vector_tuples.append(tups)
        
    # 3. Greedy Random Restart
    best_count = 0
    best_selection = []
    
    rng = np.random.default_rng(42)
    
    # We will run many fast restarts.
    # To speed up, we can use a list of integers for banned status (0 or 1)
    # or stick with bools. Bool list is fine.
    # The bottleneck is the inner loop.
    
    # Let's try to optimize the inner loop by using local variables.
    vt = vector_tuples
    n_vec = N
    max_ban = max_id
    
    for i in range(50000): # Increased restart count
        if time.time() - start_time > 580:
            break
            
        order = rng.permutation(n_vec)
        banned = [0] * max_ban
        count = 0
        selected_indices = []
        
        for idx in order:
            tups = vt[idx]
            # Unroll the 4 checks for speed
            if not (banned[tups[0]] or banned[tups[1]] or banned[tups[2]] or banned[tups[3]]):
                # Accept
                banned[tups[0]] = 1
                banned[tups[1]] = 1
                banned[tups[2]] = 1
                banned[tups[3]] = 1
                count += 1
                selected_indices.append(idx)
                
        if count > best_count:
            best_count = count
            best_selection = selected_indices
            
    # Reconstruct best points
    if best_selection:
        points = V[best_selection]
    else:
        # Fallback to weight 2
        points_list = []
        for i, j in combinations(range(d), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points_list.append(p)
        points = np.array(points_list)
        
    best_points = np.array(points, dtype=np.int64)
    return best_points
