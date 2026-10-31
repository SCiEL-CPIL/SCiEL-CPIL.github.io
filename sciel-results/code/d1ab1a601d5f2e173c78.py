import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    # Compute conflicts: two vectors conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to bitsets for fast manipulation? 
    # Or just use numpy boolean arrays.
    # N4 is ~5280. 5280x5280 boolean matrix is ~27MB. Manageable.
    
    best_size = 0
    best_mask = np.zeros(N4, dtype=bool)
    
    # Greedy MIS
    # We will run greedy with different orderings.
    
    # 1. Degree ordering (ascending)
    degrees = conflicts.sum(axis=1)
    order = np.argsort(degrees)
    
    def run_greedy(order):
        selected = np.zeros(N4, dtype=bool)
        excluded = np.zeros(N4, dtype=bool)
        count = 0
        # Process in given order
        # To speed up, we can check if node is already excluded.
        for v in order:
            if excluded[v]:
                continue
            # Select v
            selected[v] = True
            count += 1
            # Exclude neighbors
            excluded[conflicts[v]] = True
            # Also exclude v itself? It's already selected.
            # The loop will skip it if we check selected, but we check excluded.
            # So we should mark v as excluded to prevent re-selection? 
            # Or just rely on the fact that we iterate each node once.
            # Since we iterate `order` which is a permutation, we visit each node once.
            # So we just need to mark neighbors as excluded.
        return count, selected
    
    # Run with degree order
    count, mask = run_greedy(order)
    if count > best_size:
        best_size = count
        best_mask = mask.copy()
        
    # Run with random orderings if time permits
    rng = np.random.default_rng(42)
    iterations = 0
    while time.time() < deadline - 5:
        random_order = rng.permutation(N4)
        count, mask = run_greedy(random_order)
        if count > best_size:
            best_size = count
            best_mask = mask.copy()
        iterations += 1
        # Limit iterations to avoid infinite loop if extremely fast
        if iterations > 2000:
            break
            
    result_v4 = vectors_v4[best_mask]
    final_vectors = np.vstack([result_v4, vectors_v1])
    return final_vectors
