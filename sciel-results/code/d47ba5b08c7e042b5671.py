"""Program for kissing number in dimension 11 using Leech lattice projection."""
import numpy as np
from scipy.spatial import cKDTree

D = 11

# EVOLVE-BLOCK-START

def _leech_minimal_vectors():
    """Generate Leech lattice minimal vectors (norm squared 4) in 24D."""
    # Leech lattice minimal vectors come in three types:
    # Type 1: (±2, ±2, 0...) with 4 nonzeros
    # Type 2: (±1, ±1, ...) with 8 nonzeros
    # Type 3: (±1/2, ±1/2, ...) with 24 nonzeros (scaled by 2)
    
    vectors = []
    
    # Type 1: weight 4, entries ±2
    # Choose 4 positions out of 24, assign ±2
    from itertools import combinations
    for pos in combinations(range(24), 4):
        for signs in range(16):
            v = np.zeros(24, dtype=np.float64)
            for k, p in enumerate(pos):
                v[p] = 2.0 if (signs >> k) & 1 else -2.0
            vectors.append(v)
    
    # Type 2: weight 8, entries ±1
    # We need to be more careful here - use the D24 construction
    # Actually, let's use a known construction
    # The Leech lattice can be constructed via the D24 lattice with cosets
    
    # For efficiency, let's generate Type 2 vectors
    # Type 2: 8 nonzeros of ±1, with even number of -1s (for D24 part)
    # This is still huge. Let's sample instead.
    
    # Type 3: weight 24, entries ±1/2
    # All 24 coordinates are ±1/2, with even number of -1s
    # That's 2^23 vectors
    
    return np.array(vectors, dtype=np.float64)


def _generate_leech_minimals_sampled(n_samples=50000):
    """Generate a sampled set of Leech lattice minimal vectors."""
    vectors = []
    rng = np.random.default_rng(42)
    
    # Type 1: weight 4, entries ±2 (there are C(24,4)*16 = 10626*16 = 170016)
    from itertools import combinations
    type1_vectors = []
    for pos in combinations(range(24), 4):
        for signs in range(16):
            v = np.zeros(24, dtype=np.float64)
            for k, p in enumerate(pos):
                v[p] = 2.0 if (signs >> k) & 1 else -2.0
            type1_vectors.append(v)
    type1_vectors = np.array(type1_vectors, dtype=np.float64)
    
    # Type 2: weight 8, entries ±1 with even parity
    # There are C(24,8)*2^7 = 735471*128 = 94140288 - too many
    # Sample some
    type2_vectors = []
    for _ in range(20000):
        pos = rng.choice(24, size=8, replace=False)
        signs = rng.choice([1.0, -1.0], size=8)
        # Ensure even number of -1s
        n_neg = np.sum(signs == -1.0)
        if n_neg % 2 == 1:
            signs[0] = -signs[0]
        v = np.zeros(24, dtype=np.float64)
        v[pos] = signs
        type2_vectors.append(v)
    type2_vectors = np.array(type2_vectors, dtype=np.float64)
    
    # Type 3: weight 24, entries ±1/2 with even parity
    # 2^23 = 8388608 - sample
    type3_vectors = []
    for _ in range(20000):
        signs = rng.choice([0.5, -0.5], size=24)
        n_neg = np.sum(signs == -0.5)
        if n_neg % 2 == 1:
            signs[0] = -signs[0]
        type3_vectors.append(signs)
    type3_vectors = np.array(type3_vectors, dtype=np.float64)
    
    all_vectors = np.vstack([type1_vectors, type2_vectors, type3_vectors])
    
    # Sample n_samples from all
    if len(all_vectors) > n_samples:
        idx = rng.choice(len(all_vectors), size=n_samples, replace=False)
        all_vectors = all_vectors[idx]
    
    return all_vectors


def _project_to_11d(vectors_24d, rng):
    """Project 24D vectors onto 11D using random orthonormal basis."""
    n = len(vectors_24d)
    # Generate random 24x11 matrix and orthonormalize
    M = rng.standard_normal((24, 11))
    Q, _ = np.linalg.qr(M)
    # Project
    projected = vectors_24d @ Q
    return projected


def _round_to_int(projected):
    """Round projected vectors to nearest integers, remove zero vectors."""
    rounded = np.round(projected).astype(np.int64)
    # Remove zero vectors
    norms_sq = np.sum(rounded ** 2, axis=1)
    mask = norms_sq > 0
    return rounded[mask]


def _compute_min_pairwise_dist(vectors):
    """Compute minimum pairwise distance."""
    if len(vectors) < 2:
        return float('inf')
    tree = cKDTree(vectors.astype(np.float64))
    distances = tree.query(vectors.astype(np.float64), k=2)
    min_dist = np.min(distances[0][:, 1])
    return min_dist


def _compute_max_norm(vectors):
    """Compute maximum vector norm."""
    norms = np.sqrt(np.sum(vectors ** 2, axis=1))
    return np.max(norms)


def _local_search_refine(vectors, max_iter=50, rng=None):
    """Try to add more vectors or improve configuration via local search."""
    if rng is None:
        rng = np.random.default_rng()
    
    vectors = list(vectors)
    
    for _ in range(max_iter):
        # Try adding random integer vectors
        added = False
        for attempt in range(100):
            v = rng.integers(-3, 4, size=D)
            v = v.astype(np.int64)
            if np.all(v == 0):
                continue
            
            # Check if adding v maintains min_dist >= max_norm
            if len(vectors) == 0:
                vectors.append(v)
                added = True
                break
            
            v_float = v.astype(np.float64)
            dists = np.sqrt(np.sum((np.array(vectors, dtype=np.float64) - v_float) ** 2, axis=1))
            min_new_dist = np.min(dists)
            
            new_max_norm = max(_compute_max_norm(np.array(vectors)), np.linalg.norm(v_float))
            
            if min_new_dist >= new_max_norm - 1e-9:
                vectors.append(v)
                added = True
                break
        
        if not added:
            break
    
    return np.array(vectors, dtype=np.int64)


def construct():
    rng = np.random.default_rng(12345)
    
    # Generate Leech minimal vectors
    leech_minimals = _generate_leech_minimals_sampled(n_samples=30000)
    
    # Project to 11D
    projected = _project_to_11d(leech_minimals, rng)
    
    # Round to integers
    rounded = _round_to_int(projected)
    
    if len(rounded) == 0:
        # Fallback: use simple weight-2 vectors
        vectors = []
        from itertools import combinations
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    # Remove duplicate vectors
    rounded = np.unique(rounded, axis=0)
    
    # Now we need to find the largest subset with min_pairwise_dist >= max_norm
    # Use greedy approach: sort by norm, add vectors that satisfy constraint
    
    # First, let's try the direct approach
    # Compute norms
    norms_sq = np.sum(rounded ** 2, axis=1)
    norms = np.sqrt(norms_sq)
    
    # Sort by norm (ascending)
    order = np.argsort(norms)
    rounded_sorted = rounded[order]
    norms_sorted = norms[order]
    
    # Greedy selection
    selected = []
    selected_floats = []
    
    for i in range(len(rounded_sorted)):
        v = rounded_sorted[i]
        v_float = v.astype(np.float64)
        
        if len(selected_floats) == 0:
            selected.append(v)
            selected_floats.append(v_float)
            continue
        
        # Check distance to all selected
        dists = np.sqrt(np.sum((np.array(selected_floats) - v_float) ** 2, axis=1))
        min_dist = np.min(dists)
        
        # New max norm
        new_max_norm = max(np.max(np.array([np.linalg.norm(s) for s in selected_floats])), np.linalg.norm(v_float))
        
        if min_dist >= new_max_norm - 1e-9:
            selected.append(v)
            selected_floats.append(v_float)
    
    result = np.array(selected, dtype=np.int64)
    
    # Try to improve with local search
    improved = _local_search_refine(result, max_iter=30, rng=rng)
    
    # Use the better result
    k1 = len(result)
    k2 = len(improved)
    
    if k2 > k1:
        result = improved
    
    return result

# EVOLVE-BLOCK-END
