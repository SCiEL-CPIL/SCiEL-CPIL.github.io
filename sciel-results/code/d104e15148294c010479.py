import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 145.0
    D = 11
    
    # Generate all integer vectors with L2 norm squared <= 5
    # This includes:
    # Norm^2 = 1: 11 * 2 = 22
    # Norm^2 = 2: C(11,2) * 4 = 55 * 4 = 220
    # Norm^2 = 4: C(11,4) * 16 = 330 * 16 = 5280
    # Norm^2 = 5: (11*10*4) [one 2, one 1] + (C(11,5)*32) [five 1s] = 440 + 14784 = 15224
    # Total pool ~ 20746 vectors. This is large but manageable for precomputation.
    # However, precomputing a 20k x 20k distance matrix is 400M entries.
    # 400M * 8 bytes = 3.2 GB. This might be too much memory or slow.
    # Let's stick to Norm^2 <= 4 first (5522 vectors).
    # 5522^2 = 30M entries. 30M * 8 bytes = 240 MB. This is safe.
    
    all_vectors = []
    
    # Norm^2 = 1
    for i in range(D):
        for s in [1, -1]:
            v = [0]*D
            v[i] = s
            all_vectors.append(v)
            
    # Norm^2 = 2
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0]*D
                    v[i] = si
                    v[j] = sj
                    all_vectors.append(v)
                    
    # Norm^2 = 4
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0]*D
            for pos, sign in zip(support, signs):
                v[pos] = sign
            all_vectors.append(v)
            
    all_vectors_np = np.array(all_vectors, dtype=np.int32)
    n_total = len(all_vectors_np)
    
    # Precompute norms
    norms_sq = np.sum(all_vectors_np ** 2, axis=1)
    
    # Precompute dot product matrix
    # dots[i, j] = all_vectors_np[i] . all_vectors_np[j]
    dots = all_vectors_np @ all_vectors_np.T
    
    # Precompute squared distances
    # dist_sq[i, j] = ||v_i||^2 + ||v_j||^2 - 2 * dots[i, j]
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    
    # We want to find a subset S such that for all i, j in S, i != j:
    # dist_sq[i, j] >= max_norm_sq(S)
    
    # Let M be the maximum norm squared in S.
    # The constraint is dist_sq[i, j] >= M for all pairs.
    # Since dist_sq[i, j] depends on i, j, and M depends on the set S, this is tricky.
    
    # Strategy:
    # Iterate over possible values of M (which must be one of the norms_sq values in S).
    # For a fixed M, we can only select vectors v with norm_sq[v] <= M.
    # And for any pair i, j selected, we must have dist_sq[i, j] >= M.
    # Also, at least one selected vector must have norm_sq == M (otherwise M is not the max).
    # Actually, if all selected have norm_sq < M, then the true max norm is smaller, say M'.
    # So we can just iterate over all possible M values present in the pool.
    
    # Let's group vectors by norm_sq.
    unique_norms = np.unique(norms_sq)
    
    best_set = []
    best_size = 0
    
    # To speed up, we can use a greedy approach with bitsets or just simple lists.
    # Given n_total ~ 5500, and we want to find max independent set in a graph where edges are "conflicts".
    # Conflict graph: edge (i, j) if dist_sq[i, j] < M.
    # We want max independent set.
    # Max independent set is NP-hard, but greedy + local search works well for these structured graphs.
    
    # Let's try a few different M values.
    # The best M is likely the largest norm in the pool, or close to it.
    # If we pick M = max(norms_sq), we can only pick vectors with norm_sq <= M (all of them).
    # But the constraint is dist_sq[i, j] >= M.
    
    # Let's iterate M from largest to smallest.
    # For each M, build the conflict graph (or just check compatibility on the fly).
    
    # To be efficient, let's precompute for each M, the list of "compatible" pairs?
    # That's too much memory.
    
    # Instead, let's use a greedy algorithm with random restarts for a fixed M.
    # We will try M = max(norms_sq) first, then decrease.
    
    # Actually, the kissing number bound suggests we can get quite large sets.
    # For M=4, we are looking at vectors with norm 2.
    # dist^2 >= 4.
    
    rng = np.random.default_rng(42)
    
    # We will try a few M values.
    # Let's prioritize M=4, then M=2, then M=1.
    # But maybe a mix of norms is better?
    # If we mix norms, M is the max norm.
    # So if we include a vector with norm^2=4, M must be >= 4.
    # If we include a vector with norm^2=2, M must be >= 2.
    
    # Let's just try M = 4, M = 2, M = 1 separately?
    # No, the set can contain vectors of different norms, as long as the max is M.
    # So for a fixed M, we can include any vector with norm_sq <= M.
    
    # Let's try M = 4 first.
    # Candidate pool: all vectors with norm_sq <= 4.
    # Constraint: dist_sq[i, j] >= 4 for all pairs.
    
    # Let's run a greedy + local search for M = 4.
    
    def greedy_local_search(M, max_time):
        # Filter candidates
        mask = norms_sq <= M
        candidates = np.where(mask)[0]
        
        if len(candidates) == 0:
            return []
        
        # Precompute conflict matrix for candidates?
        # n_cand ~ 5500.
        # Conflict if dist_sq[i, j] < M.
        
        best_selected = []
        best_size = 0
        
        # Multiple restarts
        n_restarts = 50
        
        for r in range(n_restarts):
            elapsed = time.time() - start_time
            if elapsed > max_time:
                break
            
            # Random order
            order = rng.permutation(candidates)
            
            selected = []
            
            # Greedy addition
            for idx in order:
                # Check compatibility with all selected
                compatible = True
                for sel in selected:
                    if dist_sq[idx, sel] < M:
                        compatible = False
                        break
                
                if compatible:
                    selected.append(idx)
            
            # Local search: try to swap or add?
            # Simple improvement: try to add any unselected vector that is compatible.
            # This is like a second pass.
            added = True
            while added:
                added = False
                for idx in candidates:
                    if idx in selected:
                        continue
                    compatible = True
                    for sel in selected:
                        if dist_sq[idx, sel] < M:
                            compatible = False
                            break
                    if compatible:
                        selected.append(idx)
                        added = True
            
            if len(selected) > best_size:
                best_size = len(selected)
                best_selected = selected
                
            # Check time
            if time.time() - start_time > max_time:
                break
        
        return best_selected
    
    # Try M = 4
    # Time budget for M=4: 100s
    sel_4 = greedy_local_search(4, 100.0)
    if len(sel_4) > best_size:
        best_size = len(sel_4)
        best_set = sel_4
        
    # Try M = 2
    # Time budget for M=2: 30s
    if time.time() - start_time < deadline:
        sel_2 = greedy_local_search(2, 30.0)
        if len(sel_2) > best_size:
            best_size = len(sel_2)
            best_set = sel_2
            
    # Try M = 1
    # Time budget for M=1: 10s
    if time.time() - start_time < deadline:
        sel_1 = greedy_local_search(1, 10.0)
        if len(sel_1) > best_size:
            best_size = len(sel_1)
            best_set = sel_1
            
    if not best_set:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    return all_vectors_np[best_set].astype(np.int64)
