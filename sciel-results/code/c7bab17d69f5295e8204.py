import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    """
    Constructs a kissing configuration in Z^11 by combining the E8 lattice
    minimal vectors (embedded in the first 8 dimensions) with D3 roots
    (in the last 3 dimensions) via orthogonal direct sum.
    
    The E8 lattice has 240 minimal vectors (norm^2 = 2).
    The D3 lattice has 12 minimal vectors (norm^2 = 2).
    
    We form the set S = { (e, 0) : e in E8_min } union { (0, d) : d in D3_min }.
    
    Each point in S has norm^2 = 2.
    
    Pairwise distances:
    - Two E8 points: dist^2 = dist_E8^2 >= 2 (min dist in E8 is 2).
    - Two D3 points: dist^2 = dist_D3^2 >= 2 (min dist in D3 is 2).
    - One E8 and one D3 point: dist^2 = 2 + 2 = 4 >= 2.
    
    So min_dist^2 = 2 and max_norm^2 = 2. Condition satisfied.
    Total points: 240 + 12 = 252.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    d = 11
    points_list = []
    
    # Generate E8 minimal vectors in 8 dimensions
    # E8 minimal vectors are:
    # 1. All vectors with two non-zero entries of +/- 1/2 and rest 0? No.
    # Actually, the standard E8 lattice minimal vectors (norm^2=2) are:
    # Type 1: 240 vectors with coordinates in {0, +/-1, +/-1/2}
    # More precisely, the 240 minimal vectors of E8 are:
    #   - 112 vectors of the form (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2) with even number of minus signs
    #   - 128 vectors of the form with two entries ±1 and rest 0
    # 
    # But wait, these have half-integer coordinates. We need integer coordinates.
    # The task requires integer points in Z^11.
    # 
    # Let me reconsider. The E8 lattice can be represented with integer coordinates
    # if we use a different basis, but the minimal vectors in the standard basis
    # have half-integers.
    #
    # Alternative: Use the D8 lattice instead? D8 has 224 minimal vectors.
    # D8 minimal vectors: all vectors with exactly two non-zero entries of ±1.
    # That's C(8,2) * 4 = 28 * 4 = 112 vectors.
    #
    # Wait, let me re-examine. The D_n lattice has 2n(n-1) minimal vectors.
    # For D8: 2*8*7 = 112.
    # For D3: 2*3*2 = 12.
    #
    # So D8 ⊕ D3 would give 112 + 12 = 124 points. That's worse than 220.
    #
    # Let me think about E8 again. Can we embed E8 in Z^8?
    # The E8 lattice is not a sublattice of Z^8 in the standard way because
    # its minimal vectors have half-integer coordinates in the standard basis.
    #
    # However, we can scale: if we multiply all coordinates by 2, we get
    # integer vectors, but then norms become 4 instead of 2.
    # 
    # Let's try: take E8 minimal vectors, multiply by 2 to get integer vectors.
    # Then each has norm^2 = 4.
    # Min distance between two such vectors: 2 * 2 = 4 (since min dist in E8 is 2,
    # after scaling by 2, min dist is 4, so dist^2 = 16? No.
    # 
    # If v is a vector in E8 with |v|^2 = 2, then 2v has |2v|^2 = 4.
    # The distance between 2v and 2w is 2|v-w|, so dist^2 = 4|v-w|^2.
    # Min |v-w|^2 for distinct minimal vectors in E8 is 2 (e.g., (1,0,...) - (0,1,...) has dist^2 = 2).
    # So min dist^2 after scaling = 4 * 2 = 8.
    # Max norm^2 = 4.
    # Condition: min_dist^2 >= max_norm^2 => 8 >= 4. Holds!
    #
    # So we can use 2 * E8_minimal_vectors in the first 8 dimensions.
    # And for D3, we can keep the standard integer vectors (norm^2 = 2).
    #
    # But wait, the norms are different: E8 part has norm^2 = 4, D3 part has norm^2 = 2.
    # Max norm^2 = 4.
    # Min dist:
    #   - Two E8-scaled: min dist^2 = 8 >= 4. OK.
    #   - Two D3: min dist^2 = 2. But 2 < 4! This fails.
    #
    # So we need all points to have the same norm, or we need to scale D3 too.
    #
    # Let's scale D3 by 2 as well: 2 * D3_minimal_vectors.
    # D3 minimal vectors have norm^2 = 2, so 2*D3 has norm^2 = 8.
    # Max norm^2 = max(4, 8) = 8.
    # Min dist:
    #   - Two E8-scaled: min dist^2 = 8 >= 8. OK (equality).
    #   - Two D3-scaled: min dist^2 = 4 * 2 = 8 >= 8. OK.
    #   - One E8-scaled, one D3-scaled: dist^2 = 4 + 8 = 12 >= 8. OK.
    #
    # Total: 240 + 12 = 252 points.
    #
    # But wait, I need to generate the E8 minimal vectors. Let me do that.
    
    # Generate E8 minimal vectors (240 vectors with norm^2 = 2)
    e8_vectors = []
    
    # Type 1: 112 vectors with two entries ±1, rest 0
    for i, j in combinations(range(8), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(8, dtype=np.int64)
                v[i] = si
                v[j] = sj
                e8_vectors.append(v)
    
    # Type 2: 128 vectors with all entries ±1/2, even number of minus signs
    # We'll generate these as integer vectors by multiplying by 2 at the end
    # So for now, we generate the half-integer vectors and store them as floats,
    # or we can generate the doubled versions directly.
    
    # Let's generate the doubled versions directly for consistency.
    # A vector (1/2, 1/2, ..., 1/2) with even number of minus signs, doubled,
    # becomes (1, 1, ..., 1) with even number of minus signs.
    
    for signs in product([1, -1], repeat=8):
        # Count minus signs
        minus_count = sum(1 for s in signs if s == -1)
        if minus_count % 2 == 0:
            v = np.array(signs, dtype=np.int64)
            e8_vectors.append(v)
    
    # Now e8_vectors contains 112 + 128 = 240 vectors.
    # But the Type 1 vectors have norm^2 = 2, and the Type 2 vectors (doubled) have norm^2 = 8.
    # This is inconsistent!
    
    # I made an error. Let me redo this.
    # 
    # The E8 minimal vectors in the standard basis are:
    #   112 vectors: two entries ±1, rest 0. Norm^2 = 2.
    #   128 vectors: all entries ±1/2, even number of minus signs. Norm^2 = 8 * (1/4) = 2.
    #
    # If I multiply everything by 2:
    #   112 vectors: two entries ±2, rest 0. Norm^2 = 8.
    #   128 vectors: all entries ±1, even number of minus signs. Norm^2 = 8.
    #
    # So all 240 vectors have norm^2 = 8 after scaling by 2.
    #
    # Let me regenerate properly.
    
    e8_vectors = []
    
    # Type 1: doubled version of (±1, ±1, 0, ...) => (±2, ±2, 0, ...)
    for i, j in combinations(range(8), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(8, dtype=np.int64)
                v[i] = 2 * si
                v[j] = 2 * sj
                e8_vectors.append(v)
    
    # Type 2: doubled version of (±1/2, ±1/2, ..., ±1/2) with even minus => (±1, ±1, ..., ±1) with even minus
    for signs in product([1, -1], repeat=8):
        minus_count = sum(1 for s in signs if s == -1)
        if minus_count % 2 == 0:
            v = np.array(signs, dtype=np.int64)
            e8_vectors.append(v)
    
    # Now all 240 vectors have norm^2 = 8.
    
    # Generate D3 minimal vectors, doubled
    # D3 minimal vectors: (±1, ±1, 0), (±1, 0, ±1), (0, ±1, ±1)
    # Doubled: (±2, ±2, 0), (±2, 0, ±2), (0, ±2, ±2)
    
    d3_vectors = []
    for i, j in combinations(range(3), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(3, dtype=np.int64)
                v[i] = 2 * si
                v[j] = 2 * sj
                d3_vectors.append(v)
    
    # Now construct the 11-dimensional points
    # E8 part: (e8_vec, 0, 0, 0) where e8_vec is in Z^8
    for e8v in e8_vectors:
        p = np.zeros(11, dtype=np.int64)
        p[:8] = e8v
        points_list.append(p)
    
    # D3 part: (0, 0, 0, d3_vec) where d3_vec is in Z^3
    for d3v in d3_vectors:
        p = np.zeros(11, dtype=np.int64)
        p[8:] = d3v
        points_list.append(p)
    
    points = np.array(points_list, dtype=np.int64)
    
    # Verify
    if points.shape[0] < 2:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    # Check no zero vectors
    if np.any(np.all(points == 0, axis=1)):
        points = points[~np.all(points == 0, axis=1)]
    
    # Verify distance condition
    k = points.shape[0]
    norms_sq = np.sum(points**2, axis=1)
    max_norm_sq = int(np.max(norms_sq))
    
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            dist_sq = int(np.sum(diff**2))
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
    
    if min_dist_sq < max_norm_sq:
        # Fallback to D11
        points_list_fb = []
        for i, j in combinations(range(11), 2):
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(11, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points_list_fb.append(p)
        points = np.array(points_list_fb, dtype=np.int64)
    
    points = np.around(points).astype(np.int64)
    return points
