import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
    # Generate all weight-4 vectors in {-1, 0, 1}^D (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
    
    # Generate all weight-1 vectors in {-2, 0, 2}^D (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
    
    # Compute dot products for conflict detection
    # Conflict if dot > 2 (since dist^2 = 8 - 2*dot >= 4 requires dot <= 2)
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Greedy Minimum-Degree MIS
    alive = np.ones(N4, dtype=bool)
    selected_indices = []
    degrees = conflicts.sum(axis=1).astype(np.int32)
    rng = np.random.default_rng(42)
    
    while np.any(alive) and time.time() < deadline:
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        
        # Find min degree among alive nodes
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[rng.integers(len(cands))]
        
        selected_indices.append(chosen)
        
        # Remove chosen and its neighbors
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        alive[to_remove] = False
        
        # Update degrees for remaining nodes
        rem = np.where(alive)[0]
        if len(rem) > 0:
            degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
        
        if time.time() > deadline - 5:
            break
    
    result_v4 = vectors_v4[selected_indices]
    
    # Combine V4 and V1 vectors
    # V1 vectors are compatible with all V4 vectors (|dot| <= 2)
    # V1 vectors are mutually compatible
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
