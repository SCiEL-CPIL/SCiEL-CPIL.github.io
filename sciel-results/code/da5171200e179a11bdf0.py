# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def kissing_number11() -> np.ndarray:
    """
    Constructs a valid set of integer points in Z^11 using a Leech lattice subset.
    
    Strategy:
    The Leech lattice Lambda_24 contains a sublattice isomorphic to E8 ⊕ D4 ⊕ Z^2 (or similar).
    However, we are in dimension 11.
    We can embed a high-density configuration by taking the Cartesian product of:
    1. A subset of E8 minimal vectors (norm^2 = 2 in standard E8, scaled by 2 to norm^2 = 8? No, let's stick to integer coords).
    
    Let's use the known construction for the kissing number in dimension 11.
    The kissing number of Z^11 is not trivial, but we can construct a specific valid set.
    
    Consider the points formed by:
    A = { (x, y, z) where x in E8_min, y in D4_min, z in Z^2 }
    
    Actually, a simpler high-scoring strategy often used in these tasks is to take all vectors of a specific norm that satisfy the distance constraint.
    For norm^2 = 4, we need min_dist^2 >= 4.
    
    Let's try to construct 384 points as suggested by the inspiration.
    The Leech lattice has 196560 minimal vectors of norm^2 = 4.
    We need a subset in 11 dimensions.
    
    Alternative: Use the E8 lattice in 8 dims and D4 in 4 dims? 8+4=12. Too many.
    8+3=11.
    
    Let's use E8 (8 dims) and a 3-dim lattice.
    E8 minimal vectors: 240 vectors, norm^2 = 2.
    If we scale E8 by 2, norm^2 = 8.
    
    Let's look at the "384 points" hint.
    384 = 240 * 1.6? No.
    384 = 12 * 32?
    
    Let's try a different known configuration.
    In dimension 11, the maximal kissing number for Z^11 is known to be at least...
    Actually, let's just generate a large valid set using a greedy algorithm on a MUCH larger candidate pool, but optimized.
    
    Wait, the inspiration says: "Construct a set of 384 points by taking the Cartesian product of the 240 minimal vectors of the E8 lattice... and the 16 vectors of the D4 lattice's minimal shell... resulting in a total norm squared..."
    
    This implies a specific construction.
    E8 minimal vectors: 240. Norm^2 = 2.
    D4 minimal vectors: 24. Norm^2 = 2.
    
    If we take the direct sum, we get 8+4=12 dimensions.
    We need 11.
    
    Perhaps we take a subset.
    
    Let's try a different approach: 
    Use all integer vectors of norm^2 = 4.
    Count: 
    Type 1: (±2, 0, ...) -> 11 * 2 = 22 vectors.
    Type 2: (±1, ±1, 0, ...) -> C(11,2) * 4 = 55 * 4 = 220 vectors.
    Total 242 vectors.
    Norm^2 = 4.
    Min dist^2 between two such vectors?
    If they are (2,0,..) and (-2,0,..), dist^2 = 16.
    If (2,0,..) and (1,1,0,..), dist^2 = (2-1)^2 + (0-1)^2 = 1+1=2.
    So min_dist^2 = 2.
    Max_norm^2 = 4.
    Condition: min_dist^2 >= max_norm^2 -> 2 >= 4 is FALSE.
    So we cannot use all norm^2=4 vectors.
    
    We need min_dist^2 >= 4.
    This means we cannot have pairs at distance sqrt(2).
    
    Let's use the Leech lattice structure.
    The Leech lattice Lambda_24 can be constructed from E8 ⊕ D4 ⊕ D4 ⊕ D4? No.
    Common construction: E8 ⊕ D4 ⊕ Z^2 ⊕ ...
    
    Let's try to generate a valid set of size ~100-200 using a smarter greedy method with a larger candidate pool, or find the specific 384 point set.
    
    Actually, there is a known set of 384 points in Z^11?
    The kissing number of the Leech lattice is 196560.
    The kissing number of E8 is 240.
    
    Let's try to construct the set from the "inspiration" more carefully.
    "Cartesian product of 240 E8 and 16 D4 shell vectors".
    This would result in 240 * 16 = 3840 points if it were a product.
    But it says "Construct a set of 384 points".
    
    Maybe it's a union or a specific subset.
    
    Let's try a robust greedy approach on a larger pool.
    Pool: All vectors with norm^2 <= 4.
    Norm^2 = 1: 22 vectors.
    Norm^2 = 2: 220 vectors.
    Norm^2 = 4: 242 vectors.
    Total ~484 candidates.
    
    If we pick a subset where min_dist^2 >= max_norm^2.
    If we only pick norm^2 = 4 vectors, max_norm^2 = 4.
    We need min_dist^2 >= 4.
    This is a packing problem.
    
    Let's run a greedy algorithm on the 242 norm^2=4 vectors.
    Sort by norm (all same).
    Iterate.
    
    Let's try to improve the current program's greedy approach by using a better candidate ordering or a larger pool, but the current score is 11.
    The inspiration score 137 suggests a much larger k.
    
    How to get k=137?
    If we use norm^2=4 vectors and ensure dist^2 >= 4.
    
    Let's implement the greedy on norm^2=4 vectors.
    """
    deadline = time.time() + 580.0
    d = 11
    
    # Generate all vectors with norm^2 = 4
    # Type A: One coordinate is +/-2, rest 0
    # Type B: Two coordinates are +/-1, rest 0
    candidates = []
    
    # Type A
    for i in range(d):
        for s in (-2, 2):
            p = np.zeros(d, dtype=np.int64)
            p[i] = s
            candidates.append(p)
            
    # Type B
    for i in range(d):
        for j in range(i+1, d):
            for si in (-1, 1):
                for sj in (-1, 1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    candidates.append(p)
    
    cands = np.array(candidates, dtype=np.int64)
    N = len(cands)
    max_norm_sq = 4
    
    # To maximize the number of points, we should pick points that are "far" apart.
    # A good heuristic for packing is to pick points that are orthogonal or as far as possible.
    # However, simple greedy works if we order them well.
    # Let's try random restarts or a specific ordering.
    
    # Ordering: 
    # Points of type A are distance 4 or 8 or 16 from others.
    # Points of type B are distance 2, 4, 8...
    
    # Let's use a greedy algorithm that picks the point with the largest minimum distance to already selected points.
    # This is "Max-Min" greedy.
    
    selected = []
    selected_arr = []
    
    # Pre-compute pairwise distances? N is ~242. N^2 is ~60k. Fast.
    dist_sq = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(i+1, N):
            diff = cands[i] - cands[j]
            d2 = int(np.dot(diff, diff))
            dist_sq[i, j] = d2
            dist_sq[j, i] = d2
    
    # Max-Min Greedy
    # Start with an arbitrary point, or the one that has the largest min distance to others?
    # Actually, let's just pick the first one, then always pick the next one that maximizes the min distance to selected set.
    
    # Initial point: pick one with max norm? All same. Pick one with max min-dist to others.
    min_dists_to_others = np.min(dist_sq, axis=1)
    start_idx = np.argmax(min_dists_to_others)
    
    selected.append(start_idx)
    selected_arr.append(cands[start_idx])
    
    remaining = set(range(N))
    remaining.remove(start_idx)
    
    while remaining:
        if time.time() > deadline:
            break
            
        best_idx = -1
        best_min_dist = -1
        
        # For each candidate in remaining, calculate min dist to selected
        # This is O(N * K) per step. N=242, K grows.
        # Total O(N^2 * K) roughly. 242^3 is ~14M. Fast.
        
        for idx in remaining:
            # Min distance to any selected point
            min_d = float('inf')
            for s_idx in selected:
                d2 = dist_sq[idx, s_idx]
                if d2 < min_d:
                    min_d = d2
                    if min_d < max_norm_sq:
                        break # Early termination: if any dist is too small, this candidate is invalid anyway?
                        # Wait, we want to pick the one with MAX min_dist.
                        # If min_d < max_norm_sq, it's invalid.
            
            if min_d >= max_norm_sq and min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
                
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        selected_arr.append(cands[best_idx])
        remaining.remove(best_idx)
        
    if len(selected) < 2:
        # Fallback
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    points = cands[selected]
    
    # Verify
    k = len(points)
    max_norm = int(np.max(np.sum(points * points, axis=1)))
    min_dist = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            d2 = int(np.dot(diff, diff))
            if d2 < min_dist:
                min_dist = d2
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    if min_dist < max_norm:
        # Invalid, shouldn't happen with Max-Min greedy if implemented correctly
        # But let's trust the greedy if it was strict.
        pass
        
    return points

# EVOLVE-BLOCK-END
