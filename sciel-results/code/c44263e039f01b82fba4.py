"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time


def _e8_vectors():
    """Return the 240 E8 lattice vectors (integer, norm 2)."""
    vecs = []
    # Type 1: permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 8
                    v[i] = si
                    v[j] = sj
                    vecs.append(tuple(v))
    # Type 2: permutations of (±1/2,)*8 -- but we need integers.
    # Actually E8 vectors of norm 2 are:
    #   (±1, ±1, 0, 0, 0, 0, 0, 0) permutations: C(8,2)*4 = 112
    #   (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2) with even number of minus signs: 128
    # The second type has half-integer coordinates. To make them integer, scale by 2.
    # But then norms become 4 instead of 2. We need a consistent scaling.
    # Let's use the standard E8 root system scaled so all coordinates are integers.
    # Multiply everything by 2: Type 1 becomes (±2, ±2, 0,...) norm^2 = 8
    # Type 2 becomes (±1, ±1, ..., ±1) with even number of minuses, norm^2 = 8
    # So scaled E8 roots have norm^2 = 8, i.e., norm = 2*sqrt(2).
    # For our task, we just need min pairwise dist >= max norm.
    # If all vectors have the same norm, then min dist >= norm means |v_i - v_j| >= |v_i| for all i != j.
    # For E8 roots (scaled), |v_i|^2 = 8, |v_i - v_j|^2 = 8 + 8 - 2*v_i.v_j = 16 - 2*v_i.v_j
    # We need 16 - 2*v_i.v_j >= 8 => v_i.v_j <= 4
    # For E8 roots, inner products are 0, ±2, ±4. Max is 4. So condition holds!
    vecs = []
    # Type 1: permutations of (2, 2, 0, 0, 0, 0, 0, 0) with signs
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 8
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vecs.append(tuple(v))
    # Type 2: (±1, ±1, ±1, ±1, ±1, ±1, ±1, ±1) with even number of minus signs
    for signs in range(256):
        # Count minus signs
        minus_count = bin(signs).count('1')
        if minus_count % 2 == 0:
            v = []
            for k in range(8):
                if (signs >> k) & 1:
                    v.append(-1)
                else:
                    v.append(1)
            vecs.append(tuple(v))
    return np.array(vecs, dtype=np.int64)


def construct():
    t0 = time.time()
    deadline = t0 + 120.0  # safety margin
    
    # E8 vectors in first 8 dimensions, padded with 3 zeros
    e8 = _e8_vectors()
    n_e8 = e8.shape[0]
    # Pad to 11 dimensions
    padded = np.zeros((n_e8, D), dtype=np.int64)
    padded[:, :8] = e8
    
    # Check validity: min pairwise dist >= max norm
    norms = np.linalg.norm(padded, axis=1)
    max_norm = np.max(norms)
    
    # Compute all pairwise distances
    # For 240 vectors, this is 240*239/2 = 28680 pairs, fast
    diff = padded[:, np.newaxis, :] - padded[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    if min_dist < max_norm:
        # Should not happen, but fallback
        # Use the seed construction
        import itertools
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    # Try to add more vectors: coordinate pair vectors in the last 3 dims or mixed
    # Actually, let's try to add vectors that use the 9th, 10th, 11th coordinates
    # A simple approach: add vectors like (0,0,0,0,0,0,0,a,b,c) where a,b,c are small integers
    # and check if they maintain the property
    
    # For now, return the E8-based construction which gives 240 vectors
    # This is better than the seed's 220
    
    if time.time() > deadline - 5.0:
        return padded
    
    # Try to extend with additional vectors
    # Add vectors supported on coordinates 8,9,10 (0-indexed)
    extra = []
    for i in range(8, D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    extra.append(v)
    
    # Also try vectors with one nonzero in last 3 coords
    for i in range(8, D):
        for si in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = si
            extra.append(v)
    
    if extra:
        extra_arr = np.array(extra, dtype=np.int64)
        # Check each extra vector against all existing vectors
        candidate = np.vstack([padded, extra_arr])
        n = candidate.shape[0]
        
        # Check norms
        norms_c = np.linalg.norm(candidate, axis=1)
        max_norm_c = np.max(norms_c)
        
        # Check pairwise distances
        # This might be slow for large n, but n is at most 240 + 9 + 6 = 255
        if n < 300:
            diff_c = candidate[:, np.newaxis, :] - candidate[np.newaxis, :, :]
            dists_c = np.linalg.norm(diff_c, axis=2)
            np.fill_diagonal(dists_c, np.inf)
            min_dist_c = np.min(dists_c)
            
            if min_dist_c >= max_norm_c:
                return candidate
    
    return padded
# EVOLVE-BLOCK-END
