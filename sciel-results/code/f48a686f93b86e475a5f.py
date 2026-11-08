import numpy as np
from itertools import combinations
import time

D = 11

def _generate_weight_k_vectors(k):
    """Generate all integer vectors in R^D with exactly k non-zero entries, each +/- 1."""
    vectors = []
    for pos in combinations(range(D), k):
        for signs in range(2**k):
            v = np.zeros(D, dtype=np.int64)
            for i, p in enumerate(pos):
                v[p] = 1 if (signs >> i) & 1 else -1
            vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def _greedy_pack(vectors, max_dot, n_trials=20):
    """
    Greedily select a subset of vectors such that pairwise dot product <= max_dot.
    """
    if len(vectors) == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    N = len(vectors)
    dots = vectors @ vectors.T
    
    best_indices = []
    rng = np.random.default_rng(12345)
    
    for trial in range(n_trials):
        perm = rng.permutation(N)
        current_indices = []
        
        for idx in perm:
            if len(current_indices) > 0:
                if np.any(dots[idx, current_indices] > max_dot):
                    continue
            current_indices.append(idx)
                
        if len(current_indices) > len(best_indices):
            best_indices = current_indices
            
    return vectors[best_indices].copy()

def _refine_greedy(vectors, max_dot, n_restarts=10):
    """
    Multiple random shuffles for greedy packing.
    """
    if len(vectors) == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    N = len(vectors)
    dots = vectors @ vectors.T
    
    best_indices = []
    rng = np.random.default_rng(999)
    
    for _ in range(n_restarts):
        perm = rng.permutation(N)
        current = []
        for idx in perm:
            if len(current) > 0 and np.any(dots[idx, current] > max_dot):
                continue
            current.append(idx)
        if len(current) > len(best_indices):
            best_indices = current
            
    return vectors[best_indices].copy()

def _leech_projection_pack():
    """
    Attempt to project Leech lattice minimal vectors onto 11D and round to integers.
    This is a heuristic approach that may or may not yield good packings.
    """
    # Leech lattice minimal vectors have norm squared 4 in 24D
    # We'll generate a random orthonormal basis for 11D subspace of 24D
    # and project, then round to nearest integer
    
    # For efficiency, we'll work with a subset of Leech minimal vectors
    # The Leech lattice has 196,560 minimal vectors, but we'll generate a smaller set
    
    # Generate random orthonormal basis via QR decomposition
    rng = np.random.default_rng(42)
    M = rng.standard_normal((24, D))
    Q, R = np.linalg.qr(M)
    Q = Q[:, :D]  # 24 x D orthonormal basis
    
    # Generate a subset of Leech minimal vectors
    # Leech minimal vectors: all vectors with entries in {0, ±1/2, ±1} 
    # with norm squared 4. A simple subset: weight 4 vectors with entries ±1
    # But in 24D, we can do better. Let's use weight 4 vectors in 24D.
    
    # Actually, let's just generate random integer vectors and see if projection helps
    # This is likely to be less effective than direct greedy, but let's try
    
    # Generate random integer vectors in 11D with entries in {-1, 0, 1}
    n_candidates = 5000
    candidates = rng.integers(-1, 2, size=(n_candidates, D))
    
    # Remove zero vectors
    mask = np.any(candidates != 0, axis=1)
    candidates = candidates[mask]
    
    if len(candidates) == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    # For each candidate, compute its norm squared
    norms_sq = np.sum(candidates**2, axis=1)
    
    # We want vectors with the same norm squared for fair comparison
    # Let's group by norm squared and pick the most common
    unique_norms, counts = np.unique(norms_sq, return_counts=True)
    if len(unique_norms) == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    # Pick the norm squared that has the most vectors
    target_norm_sq = unique_norms[np.argmax(counts)]
    mask = norms_sq == target_norm_sq
    candidates = candidates[mask]
    
    if len(candidates) == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    # Condition: dist^2 >= norm_sq => 2*norm_sq - 2*dot >= norm_sq => dot <= norm_sq/2
    max_dot = target_norm_sq // 2
    
    # Greedy pack
    packed = _refine_greedy(candidates, max_dot, n_restarts=20)
    
    return packed

def construct():
    start_time = time.time()
    timeout = 115  # Leave 5s margin
    
    best_vectors = np.array([], dtype=np.int64).reshape(0, D)
    best_k = 0
    
    # 1. Weight 2 vectors - all compatible
    v2 = _generate_weight_k_vectors(2)
    if len(v2) > best_k:
        best_vectors = v2
        best_k = len(v2)
    
    if time.time() - start_time > timeout:
        return best_vectors

    # 2. Weight 3 vectors
    v3 = _generate_weight_k_vectors(3)
    packed_v3 = _refine_greedy(v3, max_dot=1, n_restarts=10)
    if len(packed_v3) > best_k:
        best_vectors = packed_v3
        best_k = len(packed_v3)
    
    if time.time() - start_time > timeout:
        return best_vectors

    # 3. Weight 4 vectors
    v4 = _generate_weight_k_vectors(4)
    packed_v4 = _refine_greedy(v4, max_dot=2, n_restarts=15)
    if len(packed_v4) > best_k:
        best_vectors = packed_v4
        best_k = len(packed_v4)
    
    if time.time() - start_time > timeout:
        return best_vectors

    # 4. Try Leech projection heuristic
    if time.time() - start_time < timeout - 20:
        leech_packed = _leech_projection_pack()
        if len(leech_packed) > best_k:
            best_vectors = leech_packed
            best_k = len(leech_packed)
    
    # 5. More trials for weight 4 if time allows
    if time.time() - start_time < timeout - 10:
        packed_v4_more = _refine_greedy(v4, max_dot=2, n_restarts=30)
        if len(packed_v4_more) > best_k:
            best_vectors = packed_v4_more
            best_k = len(packed_v4_more)

    return best_vectors
