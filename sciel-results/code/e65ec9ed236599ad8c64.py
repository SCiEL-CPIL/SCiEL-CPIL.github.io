# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Returns a 2-D array of shape (k, 11) with integer entries (int64) such that:
      - second dimension is 11
      - k >= 2
      - no point is the zero vector
      - min squared pairwise distance >= max squared norm
    
    Strategy:
    1. Generate pool of weight-w vectors for w=4 and w=5.
       For w=4: Norm^2=4, Constraint dot <= 2. Pool size 5280.
       For w=5: Norm^2=5, Constraint dot <= 2. Pool size 14784.
       
    2. Use a fast greedy selection with bitset conflict masks.
       Precompute conflict masks for each vector: mask[i] is a bitset of indices j
       such that dot(pool[i], pool[j]) > max_dot.
       Then greedy selection: maintain a bitset of 'available' vectors.
       When we select vector i, we remove i and all its conflicts from 'available'.
       
       This is O(N^2 / 64) for mask generation + O(N * N/64) for selection (approx).
       Actually, selection is O(N * (N/64)) bit operations if we update a global bitset.
       
       Let's refine:
       Precompute D = pool @ pool.T.
       For each i, conflicts[i] = np.where(D[i] > max_dot)[0] (excluding i itself if D[i,i]=w > max_dot, which is true).
       
       Greedy with bitsets:
       available = np.ones(N, dtype=bool)
       selected = []
       # We need an ordering. Random permutation is good.
       # But updating 'available' by removing conflicts of a selected item is O(N) per item.
       # Total O(k * N).
       
       Alternative:
       Use a priority queue or just iterate through a shuffled list.
       Keep a set of 'blocked' indices.
       
       Let's stick to the numpy boolean mask approach.
       available = np.ones(N, dtype=bool)
       for idx in shuffled_order:
           if not available[idx]:
               continue
           # Select idx
           selected.append(idx)
           # Block idx and its conflicts
           # conflicts_mask[idx] is a boolean array of length N
           available[conflicts_mask[idx]] = False
           
       This is efficient if conflicts_mask is precomputed as a list of boolean arrays or a 2D boolean matrix.
       2D boolean matrix N x N: 5280^2 ~ 27M booleans ~ 27MB. 14784^2 ~ 218M booleans ~ 218MB.
       This is feasible.
       
       Steps:
       1. Generate pool.
       2. Compute D.
       3. Compute conflict_matrix = (D > max_dot).
       4. Run multiple random shuffles.
       5. Track best set.
       
       We can also try to improve by starting with a high-density core?
       Random shuffle is usually decent for this type of packing.
       
       Let's implement this.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    d = 11
    
    def generate_pool(w):
        pool = []
        for idx in itertools.combinations(range(d), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = np.zeros(d, dtype=np.int64)
                for i, s in zip(idx, signs):
                    v[i] = s
                pool.append(v)
        return np.array(pool)
    
    def greedy_bitset(pool, max_dot, time_limit, n_restarts=50):
        n = len(pool)
        if n == 0:
            return np.empty((0, d), dtype=np.int64)
        
        # Precompute dot products
        D = pool @ pool.T
        
        # Conflict matrix: True if dot > max_dot
        # Note: D[i,i] = w. Since max_dot < w (for w=4, max_dot=2, w=4), self is always a conflict.
        # We want to select distinct vectors, so self-conflict is fine (we just don't pick same index twice).
        conflict_matrix = D > max_dot
        
        best_set = []
        np.random.seed(42)
        
        for restart in range(n_restarts):
            if time.time() > time_limit:
                break
            
            order = np.random.permutation(n)
            available = np.ones(n, dtype=bool)
            selected = []
            
            for idx in order:
                if time.time() > time_limit:
                    break
                
                if not available[idx]:
                    continue
                
                # Select idx
                selected.append(idx)
                
                # Update available: block idx and all its conflicts
                available[conflict_matrix[idx]] = False
            
            if len(selected) > len(best_set):
                best_set = selected
                # Early exit if we hit a theoretical limit? No obvious limit.
                # If we find a very large set, we can stop early if time is tight?
                # No, let's just keep improving.
        
        if len(best_set) == 0:
            return np.empty((0, d), dtype=np.int64)
        
        return pool[best_set]

    # Try weight 4
    pool4 = generate_pool(4)
    result4 = greedy_bitset(pool4, 2, deadline, n_restarts=100)
    k4 = len(result4)
    
    # Try weight 5
    pool5 = generate_pool(5)
    result5 = greedy_bitset(pool5, 2, deadline, n_restarts=50)
    k5 = len(result5)
    
    # Weight 2 fallback
    # Points: all weight 2 vectors
    points2 = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in (-1, 1):
                for sj in (-1, 1):
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points2.append(v)
    arr2 = np.array(points2, dtype=np.int64)
    k2 = len(arr2) # 220
    
    # Select the best result
    if k4 >= k5 and k4 >= k2:
        arr = result4
    elif k5 >= k4 and k5 >= k2:
        arr = result5
    else:
        arr = arr2
        
    arr = np.around(arr).astype(np.int64)
    return arr

# EVOLVE-BLOCK-END
