# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points such that the minimum
    squared distance between any two points is >= the maximum squared norm of any point.
    
    Structural change: instead of the previous union-of-E8-and-D3 approach (252 pts, norm^2=8),
    we use a direct-sum construction based on D4 x D3 x D4 (or similar) with a different
    scaling, and then do a much more aggressive greedy expansion using all candidate vectors
    with norm^2 <= max_norm^2, checking them in order of increasing norm.
    
    Key insight: if we can find a base configuration where min_dist^2 == max_norm^2 == m,
    then we can add any new point p with ||p||^2 <= m and ||p - v||^2 >= m for all existing v.
    
    Alternative structural approach: use the Leech lattice projection or a different
    lattice construction. But for integer points in Z^11, let's try:
    
    Approach A: Use all vectors of norm^2 = 4 with a specific structure.
    There are C(11,2)*4 = 220 vectors of type (±2,±2,0,...) -- wait, that's norm^2=8.
    Vectors of norm^2=4: (±2,0,...,0): 22 vectors; (±1,±1,0,...): C(11,2)*4=220 vectors.
    Total: 242 vectors of norm^2=4.
    Min distance between two such vectors: if they share a coordinate, could be small.
    E.g., (2,0,...) and (2,2,0,...) have dist^2 = 0+4 = 4. So min_dist^2 = 4 = max_norm^2. Valid!
    But can we add more? A vector of norm^2=4 like (1,1,0,...) vs (2,0,...): dist^2 = 1+1=2 < 4. Invalid.
    So we can't mix all of them.
    
    Approach B: Use a subset of norm^2=4 vectors that are mutually at distance >= 4.
    The 22 vectors (±2,0,...,0) are mutually at distance 4 or 8. 
    The 220 vectors (±1,±1,0,...) -- two of these can have dist^2 = 0 (same), 2, 4, or 8.
    dist^2=2 when they share one coordinate with opposite sign: (1,1,0) and (1,-1,0) -> (0,2,0) -> dist^2=4. Wait: (1,1)-(1,-1) = (0,2), norm^2=4. OK.
    (1,1,0) and (1,0,1) -> (0,1,-1), norm^2=2. So dist^2=2 < 4. Invalid.
    
    So we need to be careful. Let's use the following structural idea:
    
    Use the D4 lattice in 4 dimensions, D3 in 3, and D4 in 4, but as a DIRECT SUM with appropriate scaling.
    D4 roots (norm^2=2): 24 vectors. Scaled by 2: norm^2=8, 24 vectors.
    D3 roots (norm^2=2): 12 vectors. Scaled by 2: norm^2=8, 12 vectors.
    
    Direct sum D4 x D3 x D4: 24*12*24 = 6912 points, each with norm^2 = 8+8+8 = 24.
    Min distance: if only D4 component differs, dist^2 >= 8 (min in scaled D4). But 8 < 24. Invalid.
    
    So direct sum doesn't work directly. We need min_dist^2 >= max_norm^2.
    
    Let me reconsider. The constraint is: for all distinct pairs (u,v), ||u-v||^2 >= max(||u||^2, ||v||^2) for the whole set.
    If all points have the same norm^2 = m, then we need min_dist^2 >= m.
    
    For a set of points all with norm^2 = m, the condition is min_{u≠v} ||u-v||^2 >= m.
    This is equivalent to: for all u≠v, ||u||^2 + ||v||^2 - 2<u,v> >= m, i.e., 2m - 2<u,v> >= m, i.e., <u,v> <= m/2.
    
    So we need a set of integer vectors in Z^11, all with norm^2 = m, such that the maximum inner product between any two distinct vectors is <= m/2.
    
    For m=8: need <u,v> <= 4.
    For m=4: need <u,v> <= 2.
    For m=2: need <u,v> <= 1.
    
    Let's try m=4. All vectors with norm^2=4 in Z^11.
    Types:
    - (±2, 0, ..., 0): 22 vectors. Inner product between two: 0 or 4. If same sign, <u,v>=4 > 2. Invalid pair.
      So we can take at most one from each {±e_i} pair. 11 vectors.
    - (±1, ±1, 0, ..., 0): C(11,2)*4 = 220 vectors.
      Inner product between two such vectors: can be -2, 0, 2.
      If they share no coordinates: 0.
      If they share one coordinate: ±1*±1 = ±1. Wait, each has exactly two ±1's. If they share one position, the inner product is ±1.
      If they share both positions: ±1*±1 ± 1*±1 = -2, 0, or 2.
      So max inner product = 2. We need <= 2. So all pairs of type (±1,±1) are OK with each other!
      But with type (±2,0,...): (2,0,...) and (1,1,0,...) have inner product 2. OK (<=2).
      (2,0,...) and (-1,1,0,...) have inner product -2. OK.
      (2,0,...) and (1,0,1,...) have inner product 2. OK.
      So actually, ALL 22 + 220 = 242 vectors of norm^2=4 satisfy <u,v> <= 2 for all distinct pairs?
      Wait, (2,0,...) and (2,0,...) are the same vector, not distinct.
      (2,0,...) and (-2,0,...): inner product = -4. OK.
      (2,0,...) and (2,0,...) is not a distinct pair.
      So the only issue is (2,0,...) and (2,0,...) which is the same.
      
      Let me double-check: (2,0,...,0) and (1,1,0,...,0) -- inner product = 2*1 + 0*1 = 2. <= 2. OK.
      (2,0,...,0) and (-1,1,0,...,0) -- inner product = -2. OK.
      (2,0,...,0) and (1,0,1,0,...,0) -- inner product = 2. OK.
      
      So all 242 vectors of norm^2=4 in Z^11 form a valid configuration with m=4!
      k = 242.
      
      Can we add more? What about vectors of norm^2 < 4? 
      Norm^2=2: (±1,±1,0,...) -- wait, that's norm^2=2, not 4.
      (±1,±1,0,...) has norm^2=2. There are C(11,2)*4 = 220 such vectors.
      If we add a vector of norm^2=2, then max_norm^2 becomes 4 (from the others).
      We need ||p - v||^2 >= 4 for all v in the set.
      For p with norm^2=2 and v with norm^2=4: ||p-v||^2 = 2 + 4 - 2<p,v> = 6 - 2<p,v>.
      Need 6 - 2<p,v> >= 4, i.e., <p,v> <= 1.
      p = (1,1,0,...), v = (2,0,...): <p,v> = 2 > 1. Invalid.
      So we can't add norm^2=2 vectors next to norm^2=4 vectors.
      
      What about a completely different set with norm^2=2?
      All 220 vectors of norm^2=2: (±1,±1,0,...).
      Inner product between two: can be -2, 0, 2.
      Need <u,v> <= 1 (since m=2, m/2=1).
      But two vectors can have inner product 2: (1,1,0) and (1,1,0) is same. (1,1,0) and (1,1,0) -- distinct? No.
      (1,1,0) and (1,1,0) -- same vector.
      (1,1,0) and (-1,-1,0): inner product = -2. OK.
      (1,1,0) and (1,-1,0): inner product = 0. OK.
      (1,1,0) and (1,1,0) -- not distinct.
      Can two distinct norm^2=2 vectors have inner product 2? 
      (1,1,0,...) and (1,1,0,...) -- same.
      (1,1,0,...) and (1,1,0,...) -- no other way to get inner product 2 with norm^2=2 vectors.
      Because <u,v> = sum of products, each term is ±1*±1 or 0. Max is when both have 1's in same positions.
      (1,1,0) and (1,1,0) -- same.
      (1,1,0) and (1,1,0) -- only one such vector per support/sign pattern.
      So distinct vectors have inner product <= 1? 
      (1,1,0) and (1,1,0) -- same.
      (1,1,0) and (1,0,1): inner product = 1. OK.
      (1,1,0) and (0,1,1): inner product = 1. OK.
      (1,1,0) and (1,1,0) -- same.
      So yes, all distinct pairs have inner product <= 1. So all 220 vectors of norm^2=2 form a valid set with m=2.
      k = 220.
      
      But 242 > 220. So the norm^2=4 set is better.
      
      Can we do even better? What about norm^2=8?
      We need <u,v> <= 4.
      Vectors of norm^2=8 in Z^11:
      - (±2,±2,0,...): C(11,2)*4 = 220 vectors.
      - (±2,±1,±1,0,...): C(11,3)*4*4 = 165*16 = 2640 vectors. Wait, choose 3 positions, one gets ±2, two get ±1. C(11,3)*2*2*2 = 165*8 = 1320.
      - (±1,±1,±1,±1,0,...): C(11,4)*16 = 330*16 = 5280 vectors.
      - (±2,±2,±2,0,...): C(11,3)*8 = 165*8 = 1320 vectors.
      - (±2,±2,±1,±1,0,...): C(11,4)*4*4 = 330*16 = 5280 vectors.
      - etc.
      
      This is getting complex. The key question is: can we find a large set of norm^2=8 vectors with pairwise inner product <= 4?
      
      The E8 roots (240 vectors) in 8 dimensions, embedded in 11, have norm^2=8 and pairwise inner product in {-2,-1,0,1,2} (for distinct non-opposite). Max inner product = 2 <= 4. So all 240 are valid.
      Plus D3 roots (12 vectors) in 3 dimensions, embedded in 11, have norm^2=8.
      Between an E8 vector (a,0) and D3 vector (0,b): inner product = 0 <= 4. OK.
      So 240 + 12 = 252 vectors, all norm^2=8, max inner product <= 2 <= 4. Valid!
      
      Can we add more? The previous program did greedy expansion. Let's do a more thorough job.
      
      Actually, let me try a different structural idea: use the D5 lattice or other constructions.
      
      For now, let's use the 252-point base and do a very thorough greedy expansion, checking ALL candidate vectors with norm^2 <= 8, in order of increasing norm, and adding them if they satisfy the distance constraint.
    """
    start_time = time.time()
    deadline = start_time + 590.0
    
    d = 11
    
    # Build E8 minimal vectors (240 points in R^8, norm^2 = 8)
    def build_e8_minimal():
        pts = []
        for i in range(8):
            for j in range(i + 1, 8):
                for si in [2, -2]:
                    for sj in [2, -2]:
                        v = np.zeros(8, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        pts.append(v)
        for signs in itertools.product([1, -1], repeat=8):
            if sum(1 for s in signs if s == -1) % 2 == 0:
                pts.append(np.array(signs, dtype=np.int64))
        return pts
    
    # Build D3 minimal vectors (12 points in R^3, norm^2 = 8)
    def build_d3_minimal():
        pts = []
        for i in range(3):
            for j in range(i + 1, 3):
                for si in [2, -2]:
                    for sj in [2, -2]:
                        v = np.zeros(3, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        pts.append(v)
        return pts
    
    e8_pts = build_e8_minimal()
    d3_pts = build_d3_minimal()
    
    # Build base 252-point configuration
    base_points = []
    for v8 in e8_pts:
        v11 = np.zeros(11, dtype=np.int64)
        v11[:8] = v8
        base_points.append(v11)
    for v3 in d3_pts:
        v11 = np.zeros(11, dtype=np.int64)
        v11[8:] = v3
        base_points.append(v11)
    
    best_points = base_points[:]
    best_k = len(base_points)
    
    # Generate all candidate integer vectors with norm^2 <= 8
    # Coordinates in range [-2, 2]
    def generate_all_candidates(max_norm_sq=8, dim=11):
        candidates = []
        def gen(idx, current, norm_sq):
            if norm_sq > max_norm_sq:
                return
            if idx == dim:
                if norm_sq > 0:
                    candidates.append(current.copy())
                return
            for val in range(-2, 3):
                new_norm = norm_sq + val * val
                if new_norm > max_norm_sq:
                    continue
                current[idx] = val
                gen(idx + 1, current, new_norm)
            current[idx] = 0
        gen(0, np.zeros(dim, dtype=np.int64), 0)
        return candidates
    
    # Try greedy expansion
    if time.time() < deadline - 30:
        try:
            candidates = generate_all_candidates(8, 11)
            base_set = set(tuple(p) for p in base_points)
            candidates = [p for p in candidates if tuple(p) not in base_set]
            candidates.sort(key=lambda p: int(np.sum(p ** 2)))
            
            current_set = base_points[:]
            current_set_arr = np.array(current_set)
            
            for p in candidates:
                if time.time() > deadline - 10:
                    break
                # Check if p can be added: ||p-v||^2 >= 8 for all v in current_set
                diffs = current_set_arr - p[np.newaxis, :]
                dists_sq = np.sum(diffs ** 2, axis=1)
                if np.min(dists_sq) >= 8:
                    current_set.append(p)
                    current_set_arr = np.append(current_set_arr, p[np.newaxis, :], axis=0)
            
            if len(current_set) > best_k:
                best_points = current_set
                best_k = len(current_set)
        except Exception:
            pass
    
    # Final verification
    points_arr = np.array(best_points, dtype=np.int64)
    points_arr = np.around(points_arr).astype(np.int64)
    
    if len(points_arr) < 2 or points_arr.shape[1] != 11:
        points_arr = np.array([np.ones(11, dtype=np.int64), -np.ones(11, dtype=np.int64)])
        return points_arr
    
    zero_mask = np.all(points_arr == 0, axis=1)
    if np.any(zero_mask):
        points_arr = points_arr[~zero_mask]
    
    if len(points_arr) < 2:
        points_arr = np.array([np.ones(11, dtype=np.int64), -np.ones(11, dtype=np.int64)])
        return points_arr
    
    k = len(points_arr)
    min_dist_sq = float('inf')
    max_norm_sq = 0
    
    for i in range(k):
        norm_sq = int(np.sum(points_arr[i] ** 2))
        if norm_sq > max_norm_sq:
            max_norm_sq = norm_sq
        for j in range(i + 1, k):
            diff = points_arr[i] - points_arr[j]
            dist_sq = int(np.sum(diff ** 2))
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
    
    if min_dist_sq < max_norm_sq:
        # Fallback
        fallback = []
        for v8 in e8_pts:
            v11 = np.zeros(11, dtype=np.int64)
            v11[:8] = v8
            fallback.append(v11)
        for v3 in d3_pts:
            v11 = np.zeros(11, dtype=np.int64)
            v11[8:] = v3
            fallback.append(v11)
        points_arr = np.array(fallback, dtype=np.int64)
    
    return points_arr

# EVOLVE-BLOCK-END
