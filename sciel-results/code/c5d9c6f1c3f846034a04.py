import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Generate candidates with squared norm 4
    candidates = []

    # Type 1: One 2, rest 0: ±2e_i
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)

    # Type 2: Four 1s, rest 0: ±e_i ± e_j ± e_k ± e_l
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)

    # Compute dot product matrix M[i,j] = candidates[i] . candidates[j]
    # Constraint: for i != j, M[i,j] <= 2
    M = candidates @ candidates.T

    # Build compatibility matrix: compat[i,j] = True if M[i,j] <= 2
    # Note: M[i,i] = 4, so we must exclude self or handle it.
    # We want i != j.
    compat = (M <= 2)
    np.fill_diagonal(compat, False)

    # Conflict graph: edge exists if NOT compatible
    # A = 1 - compat (with diagonal 0)
    A = (~compat).astype(np.float64)
    np.fill_diagonal(A, 0)

    # Degree matrix
    deg = A.sum(axis=1)
    
    # Graph Laplacian L = D - A
    L = np.diag(deg) - A

    # Spectral relaxation for Max-Cut / Independent Set
    # Compute leading eigenvector of L
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        L_sparse = csr_matrix(L)
        # Get the smallest non-trivial eigenvector (Fiedler vector)
        # For finding a large independent set, we want to partition the graph
        # such that edges within each part are minimized
        k = min(2, N - 1)
        if k >= 2:
            vals, vecs = eigsh(L_sparse, k=k, which='SM')
            # Use the second eigenvector (index 1) for partitioning
            if vecs.shape[1] >= 2:
                fiedler = vecs[:, 1]
            else:
                fiedler = vecs[:, 0]
        else:
            # Fallback: use simple degree-based ordering
            fiedler = deg
    except Exception:
        # Fallback if scipy fails
        fiedler = deg

    # Partition based on sign of Fiedler vector
    # Nodes with positive sign go to one set, negative to another
    # For independent set, we want nodes that are mutually compatible
    # Let's try both partitions and take the larger valid set
    
    best_set = None
    best_count = 0
    rng = np.random.default_rng(42)

    # Strategy 1: Use spectral partitioning
    # Assign nodes based on sign of Fiedler vector
    pos_mask = fiedler >= 0
    neg_mask = ~pos_mask
    
    for mask in [pos_mask, neg_mask]:
        candidate_indices = np.where(mask)[0]
        # Check if this set is independent (all pairs compatible)
        sub_compat = compat[np.ix_(candidate_indices, candidate_indices)]
        if np.all(np.fill_diagonal(sub_compat.copy(), True)):
            count = len(candidate_indices)
            if count > best_count:
                best_count = count
                best_set = candidate_indices.tolist()

    # Strategy 2: Greedy with spectral ordering
    # Order nodes by |fiedler| value (most extreme first)
    spectral_order = np.argsort(-np.abs(fiedler), kind='stable')
    
    selected = []
    available = np.ones(N, dtype=bool)
    
    for idx in spectral_order:
        if not available[idx]:
            continue
        selected.append(idx)
        available = available & compat[idx, :]
        if not np.any(available):
            break
    
    count = len(selected)
    if count > best_count:
        best_count = count
        best_set = selected[:]

    # Strategy 3: Multiple random restarts with greedy
    num_trials = 30
    for trial in range(num_trials):
        if time.time() > deadline - 5:
            break
            
        if trial == 0:
            # Degree-based (fewest conflicts first)
            degrees = np.sum(~compat, axis=1)
            order = np.argsort(degrees, kind='stable')
        elif trial == 1:
            # Reverse degree
            degrees = np.sum(~compat, axis=1)
            order = np.argsort(-degrees, kind='stable')
        else:
            # Random perturbation
            noise = rng.random(N) * 0.1
            degrees = np.sum(~compat, axis=1)
            order = np.argsort(degrees + noise, kind='stable')

        selected = []
        available = np.ones(N, dtype=bool)
        
        for idx in order:
            if not available[idx]:
                continue
            selected.append(idx)
            available = available & compat[idx, :]
            if not np.any(available):
                break

        count = len(selected)
        if count > best_count:
            best_count = count
            best_set = selected[:]

    # If we found a set, return it
    if best_set is not None:
        return candidates[best_set].astype(np.int64)

    # Fallback: just the Type 1 vectors (22 vectors)
    return candidates[:22].astype(np.int64)
