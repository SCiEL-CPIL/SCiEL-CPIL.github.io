import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Generate candidate vectors
    # We use weight 4 vectors (4 non-zero entries, each +/- 1)
    # Norm squared = 4. Condition: dot(u, v) <= 2.
    
    indices = range(D)
    subsets = list(itertools.combinations(indices, 4))
    
    vectors = []
    for subset in subsets:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for i, s in zip(subset, signs):
                v[i] = s
            vectors.append(v)
            
    vectors = np.array(vectors, dtype=np.int8)
    n = len(vectors)
    
    # 2. Compute pairwise dot products
    # To save memory and time, we can compute in batches or use full matrix if small enough
    # 5280 x 5280 int8 matrix is ~28MB. int32 is ~112MB. Feasible.
    
    # We need to find a subset of indices such that dot(v_i, v_j) <= 2 for all i != j in subset.
    # This is the Maximum Independent Set on the graph of "conflicts" (dot > 2).
    
    # Compute dot products
    # Using matrix multiplication: V @ V.T
    dots = vectors @ vectors.T
    
    # Conflicts where dot > 2
    # Note: diagonal is 4 (self-dot), which is > 2, but we don't care about self-conflicts for MIS.
    # We create a boolean mask of conflicts.
    conflict_mask = dots > 2
    
    # Remove self-conflicts for the MIS logic (or just ignore i==j)
    np.fill_diagonal(conflict_mask, False)
    
    # 3. Greedy MIS with randomization
    best_set = []
    best_size = 0
    
    # We can run multiple random orderings
    # Time budget for search: ~100s
    search_deadline = start_time + 130.0
    
    while time.time() < search_deadline:
        # Random permutation of vertices
        perm = np.random.permutation(n)
        
        selected = []
        selected_set = set()
        
        # Greedy: iterate through perm, if v has no conflict with any selected, add it.
        # Optimization: maintain a set of "blocked" vertices?
        # Or just check against selected list.
        # Since n is 5280, and selected will be maybe ~1000-2000,
        # checking each candidate against all selected is O(N * K).
        # 5280 * 2000 = 10M ops per pass. Very fast.
        
        for i in perm:
            # Check if i conflicts with any in selected
            # conflict_mask[i] is a boolean array of size n.
            # We need to check if conflict_mask[i, selected] has any True.
            
            # Using numpy:
            # if np.any(conflict_mask[i, selected]): continue
            
            # To speed up, we can maintain a bitset or just use the mask.
            # conflict_mask is a boolean array.
            
            # Optimization: if selected is empty, add.
            if not selected:
                selected.append(i)
                continue
                
            # Check conflicts
            # conflict_mask[i] is shape (n,)
            # We want to know if any conflict_mask[i, j] is True for j in selected.
            # We can use: np.any(conflict_mask[i, selected])
            
            if np.any(conflict_mask[i, selected]):
                continue
            else:
                selected.append(i)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
            
        # If we found a very large set, we might break early?
        # Or just keep going to find even better.
        
    # 4. Construct the final array
    if best_size == 0:
        # Fallback to seed-like behavior if something goes wrong
        # But best_size should be > 0.
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    result = vectors[best_set]
    
    # Verify constraint (just in case)
    # The evaluator checks: min pairwise dist >= max norm.
    # For our vectors, norm^2 = 4. Max norm^2 = 4.
    # Dist^2 = 4 + 4 - 2 dot = 8 - 2 dot.
    # We need 8 - 2 dot >= 4 => 4 >= 2 dot => dot <= 2.
    # Our construction ensures dot <= 2 for all pairs in the set.
    
    return result.astype(np.int64)
# EVOLVE-BLOCK-END
