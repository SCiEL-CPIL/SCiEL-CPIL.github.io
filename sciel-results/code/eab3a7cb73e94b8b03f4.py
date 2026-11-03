"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Generate all integer vectors with norm squared 4
    # Types:
    # 1. 2 * e_i
    # 2. e_i + e_j + e_k + e_l (all +1)
    # 3. e_i + e_j - e_k - e_l (two +1, two -1)
    # 4. e_i - e_j - e_k - e_l (one +1, three -1) -> Same as type 3 with signs flipped?
    # Actually, just iterate over all combinations of 4 indices and all sign combinations.
    
    candidates = []
    indices = range(D)
    
    # Type 1: 2 * e_i
    for i in indices:
        v = np.zeros(D, dtype=np.int64)
        v[i] = 2
        candidates.append(v)
        v[i] = -2
        candidates.append(v)
        
    # Type 2,3,4: 4 non-zero entries, each +/- 1
    for comb in itertools.combinations(indices, 4):
        i, j, k, l = comb
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        candidates.append(v)
                        
    # Remove duplicates? The generation above might produce duplicates if we are not careful?
    # No, combinations are unique. Signs are unique.
    # However, 2*e_i is distinct from 1,1,1,1 types.
    
    n_cand = len(candidates)
    # n_cand should be 22 + 330 * 16 = 22 + 5280 = 5302
    
    # We want to select a subset S such that u . v <= 2 for all u != v in S.
    # This is equivalent to finding a Max Independent Set in the conflict graph.
    # Conflict if u . v > 2.
    
    # Since n_cand is ~5300, we can't do O(N^2) pairwise check easily in Python if it's slow,
    # but 5300^2 is ~28M operations. NumPy can handle this.
    
    # Convert to matrix
    V = np.array(candidates)
    
    # Calculate dot products
    # Dot product matrix M = V @ V.T
    # We need M[i, j] <= 2 for i != j.
    # Diagonal is 4.
    
    # To save memory and time, we can process in chunks or use a greedy approach.
    # Greedy:
    # 1. Compute degrees (number of conflicts) for each node.
    # 2. Pick node with min degree, add to S.
    # 3. Remove node and its neighbors from pool.
    # 4. Repeat.
    
    # However, computing the full 5300x5300 matrix is ~28M floats. 28M * 8 bytes = 224 MB.
    # This is acceptable.
    
    M = V @ V.T
    
    # Identify conflicts: M > 2
    # Diagonal is 4, so ignore diagonal.
    conflict_mask = (M > 2)
    np.fill_diagonal(conflict_mask, False)
    
    # Degrees
    degrees = np.sum(conflict_mask, axis=1)
    
    S_indices = []
    in_S = np.zeros(n_cand, dtype=bool)
    
    # Greedy selection
    # To speed up, we can sort by degree initially, but degrees change.
    # Simple approach: loop, find min degree among remaining, pick it.
    # This is O(N^2) which is 28M steps. Might be slow in Python.
    
    # Optimization:
    # Use a heap or just iterate.
    # Given 150s, let's try a simpler heuristic first:
    # Randomly shuffle candidates, then greedily add if compatible with all current S.
    # This is O(N * |S| * D). If |S| is large, this is slow.
    
    # Better:
    # Start with empty S.
    # Iterate through candidates in random order.
    # If compatible with all in S, add to S.
    # Then do a local search to add more.
    
    # Let's use the conflict matrix to speed up compatibility check.
    # When we add a node i, we mark all its neighbors as "blocked".
    # We maintain a count of how many "active" neighbors each node has?
    # Or simply:
    # active = np.ones(n_cand, dtype=bool)
    # S = []
    # for i in range(n_cand):
    #   if not active[i]: continue
    #   # Check if i is compatible with all in S?
    #   # We can precompute for each i, the list of neighbors.
    #   # But S grows.
    
    # Let's try a different strategy:
    # The conflict graph is sparse.
    # We can use a simple greedy on the conflict graph.
    # 1. Compute neighbors for each node.
    # 2. Sort nodes by degree.
    # 3. Iterate through sorted nodes. If node is not in S and not neighbor of any node in S, add to S.
    
    # Building neighbor lists:
    # neighbors = [np.where(conflict_mask[i])[0] for i in range(n_cand)]
    # This is expensive to build in Python.
    
    # Alternative:
    # Just use the matrix M.
    # We want to find a large set S such that M[S, S] <= 2 (off-diagonal).
    
    # Let's try a random greedy with restarts.
    best_S = []
    
    for _ in range(20): # 20 restarts
        if time.time() > deadline:
            break
        order = np.random.permutation(n_cand)
        S = []
        S_set = set()
        
        # For efficiency, we can check compatibility against S using dot products.
        # But checking against all S is O(|S|).
        
        # We can use the fact that conflicts are rare.
        # Let's just iterate.
        
        for i in order:
            v_i = V[i]
            compatible = True
            for j in S:
                if np.dot(v_i, V[j]) > 2:
                    compatible = False
                    break
            if compatible:
                S.append(i)
                S_set.add(i)
        
        if len(S) > len(best_S):
            best_S = S
            
    # Local search on best_S
    # Try to add more nodes from the remaining pool.
    if best_S:
        S_set = set(best_S)
        remaining = [i for i in range(n_cand) if i not in S_set]
        
        # Sort remaining by "compatibility" with S?
        # Just iterate and add if possible.
        for i in remaining:
            if time.time() > deadline:
                break
            v_i = V[i]
            compatible = True
            for j in best_S:
                if np.dot(v_i, V[j]) > 2:
                    compatible = False
                    break
            if compatible:
                best_S.append(i)
                S_set.add(i)
                
    result = V[best_S]
    return result

# EVOLVE-BLOCK-END
