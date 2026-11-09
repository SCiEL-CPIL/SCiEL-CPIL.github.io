# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates 
    by combining E8 roots (first 8 dims) and A3 roots (last 3 dims).
    
    Returns:
        points: np.ndarray of shape (k, 11)
    """
    d = 11
    points = []
    
    # Generate E8 roots in 8 dimensions
    # E8 roots: all vectors with two +/-1 entries (112 roots) 
    # and all vectors with one +/-2 entry (56 roots)
    e8_roots = []
    
    # Type 1: two +/-1 entries
    for i in range(8):
        for j in range(i + 1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    vec = [0] * 8
                    vec[i] = si
                    vec[j] = sj
                    e8_roots.append(tuple(vec))
    
    # Type 2: one +/-2 entry
    for i in range(8):
        for si in [1, -1]:
            vec = [0] * 8
            vec[i] = 2 * si
            e8_roots.append(tuple(vec))
    
    # Generate A3 roots in 3 dimensions
    # A3 roots: all vectors with two non-zero entries which are +/-1 and one zero,
    # where the non-zero entries have opposite signs, plus the negatives.
    # Actually, A3 roots are: (±1, ±1, 0), (±1, 0, ±1), (0, ±1, ±1) with all sign combos
    # That's 12 roots total.
    a3_roots = []
    
    # (±1, ±1, 0) - 4 roots
    for si in [1, -1]:
        for sj in [1, -1]:
            vec = (si, sj, 0)
            a3_roots.append(vec)
    
    # (±1, 0, ±1) - 4 roots
    for si in [1, -1]:
        for sk in [1, -1]:
            vec = (si, 0, sk)
            a3_roots.append(vec)
    
    # (0, ±1, ±1) - 4 roots
    for sj in [1, -1]:
        for sk in [1, -1]:
            vec = (0, sj, sk)
            a3_roots.append(vec)
    
    # Now construct the combined points:
    # 1. Pure E8 roots embedded in first 8 dims, last 3 dims are 0
    for root in e8_roots:
        vec = list(root) + [0, 0, 0]
        points.append(vec)
    
    # 2. Pure A3 roots embedded in last 3 dims, first 8 dims are 0
    for root in a3_roots:
        vec = [0] * 8 + list(root)
        points.append(vec)
    
    # 3. Combined: E8 root in first 8 dims AND A3 root in last 3 dims
    # But we need to check that the combined vector has squared norm 2
    # E8 roots have squared norm 2 or 4
    # A3 roots have squared norm 2
    # Combined: norm_sq = norm_e8_sq + norm_a3_sq
    # For norm_sq = 2, we need norm_e8_sq + norm_a3_sq = 2
    # Since norm_a3_sq = 2 always, we need norm_e8_sq = 0, which is impossible for a root
    # So no combined points with both non-zero parts work for norm 2
    
    # Let's verify: E8 type 1 roots have norm_sq = 2, type 2 have norm_sq = 4
    # A3 roots have norm_sq = 2
    # So combined would have norm_sq = 4 or 6, not 2
    
    # Therefore, we only have pure E8 and pure A3 roots
    # Total: 168 + 12 = 180 points
    
    # But wait, we can also consider the original weight-2 vectors that span all 11 dimensions
    # Let's add all vectors with exactly two non-zero entries (+/-1) in any two of the 11 positions
    # This gives C(11,2) * 4 = 55 * 4 = 220 points
    
    # Actually, let me reconsider. The E8 roots I generated include vectors like (1,1,0,...,0)
    # which are already included in the weight-2 set. Let me just use the weight-2 set
    # since it's simpler and gives 220 points.
    
    # Clear and regenerate using the simple weight-2 approach
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    vec = [0] * d
                    vec[i] = si
                    vec[j] = sj
                    points.append(vec)
    
    points = np.array(points, dtype=np.int64)
    return points

# EVOLVE-BLOCK-END
