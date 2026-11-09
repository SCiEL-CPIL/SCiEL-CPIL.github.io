import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    # Generate all weight-4 vectors
    vectors = []
    indices_list = []
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(indices, signs):
                v[idx] = s
            vectors.append(v)
            
    V = np.array(vectors, dtype=np.int8)
    n_vecs = V.shape[0]
    
    # Compute dot products
    # V is (N, 11). V @ V.T is (N, N)
    dots = V @ V.T
    
    # Conflict if dot > 2 (and i != j)
    # Diagonal is 4.
    # We want to find a large independent set in graph G where (i,j) in E if dots[i,j] > 2.
    
    # Greedy approach with random restarts?
    # Or just one greedy pass with random order.
    # Given the density, one good pass might suffice.
    
    best_size = 0
    best_indices = []
    
    # Try a few random orders to get a good independent set
    for attempt in range(10):
        if time.time() > deadline:
            break
            
        order = np.random.permutation(n_vecs)
        available = np.ones(n_vecs, dtype=bool)
        selected = []
        
        for u in order:
            if not available[u]:
                continue
                
            # Check if u conflicts with any ALREADY SELECTED vector?
            # No, standard greedy MIS:
            # If we select u, we remove u and its neighbors from 'available'.
            # But we must ensure u is compatible with all previously selected.
            # In the "remove neighbors" strategy, if v was selected earlier,
            # and u is neighbor of v, u would have been removed from available.
            # So if u is available, it is NOT a neighbor of any selected vertex.
            # Thus it is compatible.
            
            selected.append(u)
            
            # Remove u and its neighbors from available
            # Neighbors are v such that dots[u, v] > 2
            # Note: dots[u, u] = 4 > 2, so u is removed.
            neighbors = np.where(dots[u, :] > 2)[0]
            available[neighbors] = False
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_indices = selected
            # If we found a very large set, we can stop early?
            # 220 is the baseline.
            # If we find > 2000, it's likely near optimal for this family.
            if best_size > 3000:
                break

    if best_size == 0:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    result = V[best_indices]
    return result.astype(np.int64)
