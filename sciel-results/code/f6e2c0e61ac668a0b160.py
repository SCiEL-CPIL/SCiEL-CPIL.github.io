import numpy as np
import itertools
import time

D = 11

def generate_candidates():
    """
    Generate a dense set of integer vectors in R^11 with norm squared <= 5.
    Includes:
    - Weight 1: (2,0,...), (1,0,...) -- actually (1,0,..) is norm 1, (2,0,..) is norm 2.
    - Weight 2: (1,1,..), (2,1,..), (2,0,..) is weight 1.
    - Weight 3: (1,1,1,..)
    - Weight 4: (1,1,1,1,..)
    
    We will generate all vectors with entries in {-2,-1,0,1,2} and norm^2 <= 5.
    """
    candidates = []
    
    # Norm^2 = 1: (1,0,..) or (-1,0,..)
    # These are very "small" and might conflict with many things if max norm is high.
    # If max norm is sqrt(5), then (1,0,..) conflicts with anything within dist < sqrt(5).
    # d((1,0,..), (1,1,..)) = 1 < sqrt(5). So they conflict.
    # It's better to exclude low norm vectors if they don't help packing density relative to their "cost".
    # However, they can fill gaps.
    # Let's include them.
    
    # Instead of iterating all 5^11, we generate by weight and value combinations.
    
    # Type: (2, 0, ..., 0)
    for i in range(D):
        candidates.append([0]*i + [2] + [0]*(D-i-1))
        candidates.append([0]*i + [-2] + [0]*(D-i-1))
        
    # Type: (1, 0, ..., 0)
    for i in range(D):
        candidates.append([0]*i + [1] + [0]*(D-i-1))
        candidates.append([0]*i + [-1] + [0]*(D-i-1))
        
    # Type: (2, 1, 0, ...)
    for i in range(D):
        for j in range(D):
            if i == j: continue
            for s1 in [2, -2]:
                for s2 in [1, -1]:
                    v = [0]*D
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
                    
    # Type: (1, 1, 0, ...)
    for i in range(D):
        for j in range(D):
            if i == j: continue
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0]*D
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
                    
    # Type: (2, -1, 0, ...) already covered by signs above
    
    # Type: (1, 1, 1, 0, ...)
    for combo in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0]*D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
            
    # Type: (1, 1, 1, 1, 0, ...)
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0]*D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
            
    # Type: (2, 2, 0, ...) norm^2 = 8 > 5. Exclude.
    # Type: (2, 1, 1, 0, ...) norm^2 = 6 > 5. Exclude.
    
    # Remove duplicates (shouldn't be any, but just in case)
    unique_candidates = []
    seen = set()
    for v in candidates:
        t = tuple(v)
        if t not in seen:
            seen.add(t)
            unique_candidates.append(v)
            
    return np.array(unique_candidates, dtype=np.int8)

def greedy_mis(conflicts, deadline, rng):
    """
    Greedy Maximum Independent Set on a graph defined by boolean conflict matrix.
    """
    N = len(conflicts)
    alive = np.ones(N, dtype=bool)
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    selected_indices = []
    
    while np.any(alive):
        if time.time() > deadline:
            break
        
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
            
        # Find min degree among alive
        min_deg = np.min(degrees[alive_indices])
        candidates = alive_indices[degrees[alive_indices] == min_deg]
        
        # Random tie-breaking
        chosen = candidates[rng.integers(len(candidates))]
        selected_indices.append(chosen)
        
        # Find neighbors to remove
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        # Update alive
        alive[to_remove] = False
        
        # Update degrees for remaining alive nodes
        remaining_alive = np.where(alive)[0]
        if len(remaining_alive) > 0:
            removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
            degrees[remaining_alive] -= removed_conflicts

    return selected_indices

def construct():
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    # Generate candidates
    vectors = generate_candidates()
    N = len(vectors)
    
    if N == 0:
        return np.empty((0, D), dtype=np.int64)
        
    # Compute dot products
    # d(u,v)^2 = ||u||^2 + ||v||^2 - 2 u.v
    # We need d(u,v)^2 < max(||u||^2, ||v||^2)
    # Let M = max(||u||^2, ||v||^2)
    # Conflict if ||u||^2 + ||v||^2 - 2 u.v < M
    # 2 u.v > ||u||^2 + ||v||^2 - M
    
    norms_sq = np.sum(vectors**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Vectorized conflict check
    # dots[i,j] = u_i . v_j
    dots = vectors @ vectors.T
    
    # u_norm_sq[i] is ||u_i||^2
    # v_norm_sq[j] is ||v_j||^2
    # M[i,j] = max(u_norm_sq[i], v_norm_sq[j])
    
    # To avoid N^2 memory for M, we can compute conflict condition directly?
    # Conflict if dots[i,j] < (norms_sq[i] + norms_sq[j] - max(norms_sq[i], norms_sq[j])) / 2
    # RHS = min(norms_sq[i], norms_sq[j]) / 2
    
    # So conflict if 2 * dots[i,j] < min(norms_sq[i], norms_sq[j])
    
    # min_norms[i,j] = min(norms_sq[i], norms_sq[j])
    # This is an outer product of sorts, but with min.
    
    # We can compute this efficiently.
    # min(u, v) = (u + v - |u - v|) / 2
    # |u - v| is element-wise absolute difference of outer products.
    
    # Let's just compute the threshold matrix.
    # N is ~7000. N^2 is 50M.
    # norms_sq[:, None] and norms_sq[None, :]
    
    u_norms = norms_sq[:, None]
    v_norms = norms_sq[None, :]
    
    # min_norms = np.minimum(u_norms, v_norms)
    # This creates a 7000x7000 array of int32/int64. ~200MB.
    # Then 2*dots < min_norms
    
    # Let's do it in chunks if N is large, but 7000 is fine.
    
    min_norms = np.minimum(u_norms, v_norms)
    
    # Conflict if 2 * dots < min_norms
    # Note: dots are integers.
    
    conflicts = (2 * dots) < min_norms
    
    # Exclude self-conflicts (diagonal)
    np.fill_diagonal(conflicts, False)
    
    best_set = []
    rng = np.random.default_rng(42)
    
    # Run multiple restarts
    while time.time() < deadline:
        current_set = greedy_mis(conflicts, deadline, rng)
        if len(current_set) > len(best_set):
            best_set = current_set
            
    if not best_set:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    result_vectors = vectors[best_set]
    return result_vectors.astype(np.int64)
