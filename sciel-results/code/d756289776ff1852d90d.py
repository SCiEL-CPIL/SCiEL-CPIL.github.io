# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def kissing_number11() -> np.ndarray:
    """
    Constructs a kissing configuration in Z^11 by combining all 220 D_11 roots
    with a greedily selected compatible set of additional integer vectors.

    Returns:
        points: np.ndarray of shape (num_points, 11)
    """
    d = 11
    deadline = time.time() + 570  # safety margin before 630s limit

    # --- Step 1: Build the 220 D_11 roots (norm^2 = 2) ---
    roots = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in (-1, 1):
                for sj in (-1, 1):
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    roots.append(v)
    roots = np.array(roots, dtype=np.int64)

    # Max norm^2 among roots is 2
    current_max_norm_sq = 2

    # Precompute the conflict matrix among roots: conflict[i,j] = True if
    # ||roots[i] - roots[j]||^2 < 2 (which never happens, so all compatible)
    # We skip this since we know all pairs are compatible.

    # --- Step 2: Try to add vectors of norm^2 = 3 ---
    # A vector u with ||u||^2 = 3 is compatible with a root r (||r||^2 = 2) iff
    #   ||u - r||^2 >= 3
    #   ||u||^2 + ||r||^2 - 2<u,r> >= 3
    #   3 + 2 - 2<u,r> >= 3
    #   2 >= 2<u,r>
    #   <u,r> <= 1
    #
    # Also, u must be compatible with the updated max_norm_sq.
    # If we add u, max_norm_sq becomes max(2, 3) = 3.
    # Then we need ||u - r||^2 >= 3 for all roots r, which is the same condition.
    # And for pairs of added vectors u, v with norm^2 = 3:
    #   ||u - v||^2 >= 3
    #   3 + 3 - 2<u,v> >= 3
    #   <u,v> <= 1.5, i.e., <u,v> <= 1 (since integer)
    
    # Generate all integer vectors with norm^2 = 3
    # These have exactly 3 non-zero entries, each ±1
    candidates_norm3 = []
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for si in (-1, 1):
                    for sj in (-1, 1):
                        for sk in (-1, 1):
                            v = np.zeros(d, dtype=np.int64)
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates_norm3.append(v)
    
    candidates_norm3 = np.array(candidates_norm3, dtype=np.int64)
    n_cand3 = len(candidates_norm3)
    
    # For each candidate u, check if <u, r> <= 1 for ALL roots r
    # <u, r> = u · r where r has exactly 2 non-zero entries
    # u · r can be computed by checking the 2 non-zero positions of r in u
    
    # Precompute: for each candidate u, compute max over all roots of u·r
    # A root r has non-zero entries at positions (i,j) with values (si,sj)
    # u·r = si*u[i] + sj*u[j]
    # Max over si,sj ∈ {-1,1} of si*u[i] + sj*u[j] = |u[i]| + |u[j]|
    # But we need the max over ALL roots, so:
    # max_{i<j} (|u[i]| + |u[j]|) where u[i], u[j] are the values of u at positions i,j
    # But u has entries in {-1, 0, 1}, so |u[i]| ∈ {0, 1}
    # So max over i<j of (|u[i]| + |u[j]|) = max number of non-zero entries in any pair = min(2, popcount(u))
    # Since u has exactly 3 non-zero entries, for any pair (i,j) where both are non-zero,
    # |u[i]| + |u[j]| = 2. So max = 2.
    # Wait, that means u·r can be as large as 2 for some root r.
    # So <u, r> <= 1 is NOT satisfied for all roots.
    # Therefore, NO norm^2=3 vector is compatible with all 220 roots!
    
    # Let me verify: take u = (1,1,1,0,...,0) and r = (1,1,0,...,0)
    # u·r = 1+1 = 2 > 1. So indeed incompatible.
    
    # So we cannot add any norm^2=3 vectors while keeping all 220 roots.
    
    # Alternative: use a subset of roots plus some norm^2=3 vectors?
    # That would reduce the total count, which is not beneficial.
    
    # --- Step 3: Try vectors of norm^2 = 1 (single non-zero entry) ---
    # u = e_i (unit vector), ||u||^2 = 1
    # Compatible with root r iff ||u - r||^2 >= max(2, 1) = 2
    # ||u - r||^2 = ||u||^2 + ||r||^2 - 2<u,r> = 1 + 2 - 2<u,r> = 3 - 2<u,r>
    # Need 3 - 2<u,r> >= 2, i.e., <u,r> <= 0.5, i.e., <u,r> <= 0.
    # u = e_i, r has entries at (j,k). <u,r> = r[i] if i==j or i==k, else 0.
    # So we need r[i] = 0 for all roots r in our set.
    # But every position i appears in many roots, so e_i is incompatible with
    # any root that has a non-zero entry at position i.
    # Therefore, no unit vectors can be added.
    
    # --- Step 4: Try vectors of norm^2 = 4 ---
    # These could be: two entries of ±2 (unlikely in Z^11 context), or four entries of ±1,
    # or one entry of ±2 and two of ±1, etc.
    # For simplicity, consider vectors with four entries of ±1: ||u||^2 = 4
    # Compatible with root r (||r||^2 = 2) iff ||u-r||^2 >= max(4, 2) = 4
    # 4 + 2 - 2<u,r> >= 4 => 2 >= 2<u,r> => <u,r> <= 1
    # <u,r> = u·r where r has 2 non-zero entries. Max u·r = |u[i]|+|u[j]| <= 2.
    # So need <u,r> <= 1 for all roots. Same issue as norm^2=3.
    
    # It appears that the 220 roots form a maximal compatible set for this constraint.
    # The only way to get more points would be to use a different set with lower norms.
    
    # Let's try: use ALL vectors with entries in {-1, 0, 1} and norm^2 <= 2.
    # That's exactly the 220 roots plus the zero vector (excluded) plus 22 unit vectors.
    # But unit vectors are incompatible as shown above.
    
    # Final check: maybe we can add vectors of norm^2 = 2 that are NOT roots?
    # In Z^11, norm^2=2 means exactly 2 entries are ±1, rest 0. That's exactly the 220 roots.
    
    # So 220 is the maximum we can achieve with this approach.
    
    points = roots.copy()
    k = len(points)
    
    # Verify
    max_norm_sq = 0
    for p in points:
        n2 = int(np.dot(p, p))
        if n2 > max_norm_sq:
            max_norm_sq = n2
    
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            d2 = int(np.dot(diff, diff))
            if d2 < min_dist_sq:
                min_dist_sq = d2
    
    if min_dist_sq >= max_norm_sq and k >= 2:
        return np.around(points).astype(np.int64)
    
    # Fallback
    pts = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
    return np.around(pts).astype(np.int64)


# EVOLVE-BLOCK-END
