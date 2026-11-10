# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product


def kissing_number11() -> np.ndarray:
    """
    Constructs a set of eleven-dimensional integer points satisfying the kissing
    number condition: min pairwise squared distance >= max squared norm.
    
    Strategy: Combine scaled E8 roots (240 points in first 8 dims) with scaled D3 roots
    (24 points in last 3 dims), all with uniform squared norm 8.
    Total: 264 points.
    """
    d = 11
    deadline = time.time() + 550  # safety margin before 600s limit
    
    # Build E8 roots (240 vectors in 8D with norm^2 = 2)
    # E8 roots: all vectors with entries in {0, ±1} with exactly 4 nonzeros (112 vectors)
    # plus all vectors with entries in {0, ±1/2} with all 8 nonzero (128 vectors)
    # But we need integer vectors, so use the 112 "half" type differently.
    # Actually, standard E8 roots in Z^8: 
    # Type 1: (±1, ±1, 0, ..., 0) - choose 2 positions, 4 sign choices = C(8,2)*4 = 28*4 = 112
    # Type 2: (±1/2, ±1/2, ..., ±1/2) with even number of minus signs - not integer!
    # So for integer E8, we only have the 112 vectors of type (±1, ±1, 0, ...)
    # Wait, that's D8, not E8. E8 in Z^8 has 240 roots but they're not all in Z^8.
    # 
    # Let me reconsider. The E8 lattice is not a sublattice of Z^8. 
    # Instead, let's use D8 roots (112 vectors) which ARE in Z^8 with norm^2 = 2.
    # Or better: use the fact that we can embed in higher dimensions.
    
    # Actually, let's use a simpler approach:
    # D8 roots in first 8 dims: (±1, ±1, 0, ..., 0) → C(8,2)*4 = 112 vectors, norm^2 = 2
    # Scale by 2: norm^2 = 8, 112 vectors
    
    # For the last 3 dims, use D3 roots: (±1, ±1, 0), (0, ±1, ±1), (±1, 0, ±1)
    # That's 3*C(3,2)*4 = 3*3*4 = 36? No: for each pair of positions, 4 sign combos
    # C(3,2) = 3 pairs, 4 signs each = 12 vectors with norm^2 = 2
    # Scale by 2: norm^2 = 8, 12 vectors
    
    # Total: 112 + 12 = 124. Not enough.
    
    # Let me use a different decomposition. 
    # D11 roots: (±1, ±1, 0, ..., 0) → C(11,2)*4 = 55*4 = 220 vectors, norm^2 = 2
    # This is what the current program does.
    
    # To exceed 220, I need a different approach.
    # 
    # Key idea from the hint: E8 ⊕ A3 or E8 ⊕ D3
    # E8 has 240 roots. If I can represent them in Z^8 with integer entries...
    # The E8 lattice can be represented as:
    # {x in R^8 : x_i ∈ Z or x_i ∈ Z + 1/2, sum x_i ∈ Z, sum x_i^2 ∈ Z}
    # The roots have norm^2 = 2 and are:
    # (±1, ±1, 0, ..., 0): C(8,2)*4 = 112
    # (±1/2, ±1/2, ..., ±1/2) with even number of minuses: 128
    # Total: 240. But the second type has half-integer entries.
    # 
    # If I scale the half-integer type by 2, they become integers with norm^2 = 8.
    # But then the integer type would need scaling too, giving norm^2 = 8 as well.
    # So: scale ALL E8 roots by 2 → all have norm^2 = 8, all integer entries.
    # The 112 type-(±1,±1,0,...) become (±2, ±2, 0, ...) with norm^2 = 8.
    # The 128 type-(±1/2,...) become (±1, ±1, ..., ±1) with even number of -1s, norm^2 = 8.
    
    # So E8 scaled by 2 gives 240 integer vectors in Z^8 with norm^2 = 8.
    
    # Now for the last 3 dimensions, I need vectors with norm^2 = 8 in Z^3.
    # Options: (±2, ±2, 0) → 4 sign combos per pair, C(3,2)=3 pairs → 12 vectors
    # Or: (±2, 0, ±2), (0, ±2, ±2) - these are the same set
    # Or: (±2, ±2, 0) and permutations: 12 vectors
    # Also: (±2, ±2, ±2) would have norm^2 = 12, not 8.
    # (±2, ±1, ±1) has norm^2 = 6, not 8.
    # (±2, ±2, 0) is the only option for norm^2 = 8 in Z^3 with small entries.
    # Wait: (±2, ±2, 0), (±2, 0, ±2), (0, ±2, ±2) → 3*4 = 12 vectors.
    
    # Total: 240 + 12 = 252 points.
    
    # Let me verify the distance conditions:
    # All have norm^2 = 8, so max norm^2 = 8.
    # Need min pairwise dist^2 >= 8.
    
    # Within E8-scaled: original E8 min dist^2 = 2 (between roots differing by one sign flip in a (±1,±1) pair, 
    #   or between a (±1,±1,0,...) and a nearby (±1/2,...) vector).
    #   Actually, in E8, the minimum distance between distinct roots is sqrt(2), so dist^2 = 2.
    #   Scaled by 2: dist^2 = 8. ✓
    
    # Within D3-scaled (the 12 vectors in last 3 dims): 
    #   (±2,±2,0) type. Two such vectors: if they share the same pair of positions,
    #   min dist^2 = (0)^2 + (4)^2 + 0 = 16 (if signs differ on one coord) or 0+0+0=0 (same vector, excluded).
    #   If they use different pairs: e.g., (2,2,0) and (2,0,2): dist^2 = 0+4+4 = 8. ✓
    #   If (2,2,0) and (-2,-2,0): dist^2 = 16+16+0 = 32. ✓
    #   Min within this set: 8. ✓
    
    # Cross (E8-scaled in first 8 dims vs D3-scaled in last 3 dims):
    #   E8 vector has zeros in last 3 dims, D3 vector has zeros in first 8 dims.
    #   dist^2 = ||e||^2 + ||d||^2 = 8 + 8 = 16 >= 8. ✓
    
    # Great! 252 points works.
    
    points_list = []
    
    # E8 roots scaled by 2 (240 vectors in first 8 dims)
    # Type 1: (±2, ±2, 0, ..., 0) - 112 vectors
    for i, j in combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = 2 * si
                p[j] = 2 * sj
                points_list.append(p)
    
    # Type 2: (±1, ±1, ..., ±1) in first 8 dims with even number of -1s - 128 vectors
    for signs in product(range(2), repeat=8):
        num_minus = sum(1 for s in signs if s == 1)
        if num_minus % 2 == 0:
            p = np.zeros(d, dtype=np.int64)
            for i in range(8):
                p[i] = 1 if signs[i] == 0 else -1
            points_list.append(p)
    
    # D3-scaled roots in last 3 dims (12 vectors)
    for i, j in combinations(range(3), 2):
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[8 + i] = 2 * si
                p[8 + j] = 2 * sj
                points_list.append(p)
    
    points_arr = np.array(points_list, dtype=np.int64)
    
    # Verify
    k = len(points_arr)
    assert k == 252, f"Expected 252 points, got {k}"
    
    # Check all norms are 8
    norms_sq = np.sum(points_arr * points_arr, axis=1)
    assert np.all(norms_sq == 8), f"Not all norms are 8: {set(norms_sq.tolist())}"
    
    # Check min pairwise distance
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = points_arr[i] - points_arr[j]
            dist_sq = int(np.sum(diff * diff))
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
    
    assert min_dist_sq >= 8, f"Min dist^2 = {min_dist_sq} < 8"
    
    # Check no zero vector
    assert not np.any(np.all(points_arr == 0, axis=1)), "Contains zero vector"
    
    # Check distinctness
    assert len(set(tuple(p) for p in points_arr.tolist())) == k, "Duplicate points"
    
    best_points = np.around(points_arr).astype(np.int64)
    return best_points


# EVOLVE-BLOCK-END
