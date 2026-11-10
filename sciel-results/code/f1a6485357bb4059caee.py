import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620  # safety margin before 630s kill
    
    d = 11
    # Generate all integer vectors with squared norm 8 (coordinates in {-2, 0, 2})
    # Each vector has exactly two non-zero coordinates, each being +/-2
    # Number of such vectors: C(11,2) * 4 = 55 * 4 = 220
    # Wait, the directive says 11,858 vectors. Let me reconsider.
    # Actually, coordinates in {-2, 0, 2} with squared norm 8 means exactly two coordinates are +/-2.
    # That gives C(11,2)*4 = 220 vectors.
    # But the directive mentions 11,858. Let me think about what pool gives that.
    # Maybe coordinates in {-1, 0, 1, 2, -2} or other combinations?
    # Squared norm 8: possible combinations of squares summing to 8:
    # - 4+4 (two coordinates of magnitude 2)
    # - 4+2+2 (one of magnitude 2, two of magnitude sqrt(2)) - not integer
    # - 1+1+1+1+1+1+1+1 (eight coordinates of magnitude 1)
    # - 4+1+1+1+1 (one of magnitude 2, four of magnitude 1)
    # - 1+1+1+1+1+1+1+1 (eight of magnitude 1)
    # - 4+4 (two of magnitude 2)
    # - 4+1+1+1+1 (one of magnitude 2, four of magnitude 1)
    # - 1+1+1+1+1+1+1+1 (eight of magnitude 1)
    # - 2+2+2+2 (four of magnitude sqrt(2)) - not integer
    # - 1+1+1+1+1+1+1+1 = 8 (eight coordinates of magnitude 1)
    # - 4+1+1+1+1 = 8 (one coordinate of magnitude 2, four of magnitude 1)
    # - 4+4 = 8 (two coordinates of magnitude 2)
    # - 1+1+1+1+1+1+1+1 = 8
    # - 4+2+2 - not integer
    # So integer vectors with squared norm 8:
    # Type A: two coordinates in {-2,2}, rest 0: C(11,2)*4 = 220
    # Type B: one coordinate in {-2,2}, four coordinates in {-1,1}, rest 0: C(11,1)*2 * C(10,4)*16 = 11*2*210*16 = 73920
    # That's way more than 11858.
    # Type C: eight coordinates in {-1,1}, rest 0: C(11,8)*256 = 165*256 = 42240
    # Total would be huge.
    
    # The directive specifically says "11,858 integer vectors with squared norm 8 (coordinates in {-2, 0, 2})"
    # With coordinates in {-2, 0, 2} only, squared norm 8 means exactly two coordinates are +/-2.
    # That's C(11,2)*4 = 220 vectors. Not 11858.
    
    # Let me just use the pool of all vectors with coordinates in {-2, -1, 0, 1, 2} and squared norm <= some bound.
    # Actually, let me re-read: the current program uses vectors with two coordinates in {-1,1}, giving squared norm 2.
    # The min squared distance between any two such vectors: if they share one coordinate position with same sign, dist^2 = 1+1=2... 
    # Actually let's compute: p has p[i]=a, p[j]=b; q has p[i]=c, p[j]=d. dist^2 = (a-c)^2 + (b-d)^2.
    # If same i,j and same signs: dist^2=0 (but points must be distinct, so this won't happen if we don't duplicate)
    # If same i, different j: dist^2 = (a-c)^2 + b^2 + d^2... this gets complex.
    
    # Let me focus on what the directive suggests: use a larger candidate pool and solve MIS.
    # I'll generate all vectors with coordinates in {-2, -1, 0, 1, 2} with squared norm exactly 4 (two coords of magnitude 1, or one of magnitude 2).
    # Actually, let me just follow the spirit: build a conflict graph and solve MIS.
    
    # For efficiency, I'll use a moderate pool. Let's use all vectors with exactly two non-zero coords, each in {-2, -1, 1, 2}.
    # That's C(11,2) * 4 * 4 = 55 * 16 = 880 vectors.
    # Squared norm ranges from 1+1=2 to 4+4=8.
    
    # Actually, let me use a simpler approach that's known to work: 
    # Use the E8 lattice construction or known kissing number constructions.
    # The kissing number in 11 dimensions is known to be 558 (from known results).
    # But we need integer points. Let me use a constructive approach.
    
    # Let me generate all vectors with coordinates in {-2,-1,0,1,2} and squared norm in {4, 8},
    # then build a conflict graph where edge exists if dist^2 < max_norm^2.
    
    # For simplicity and to meet the time budget, let me use a greedy + local search approach
    # on a well-chosen candidate set.
    
    # Candidate set: all vectors with exactly two non-zero coordinates, each in {-2, -1, 1, 2}
    candidates = []
    for i in range(d):
        for j in range(i+1, d):
            for vi in [-2, -1, 1, 2]:
                for vj in [-2, -1, 1, 2]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = vi
                    p[j] = vj
                    candidates.append(p)
    
    n = len(candidates)
    candidates = np.array(candidates)
    
    # Compute squared norms
    norms2 = np.sum(candidates**2, axis=1)
    max_norm2 = np.max(norms2)
    
    # Build conflict graph: edge between i,j if dist^2(i,j) < max_norm2
    # For efficiency, use numpy
    # dist^2(i,j) = norm2(i) + norm2(j) - 2*dot(i,j)
    # We need dist^2 < max_norm2 for all pairs in our set
    
    # Since all norms2 <= max_norm2, and we want dist^2 >= max_norm2 for non-adjacent in independent set
    # Actually we want: for selected points, min pairwise dist^2 >= max norm^2 of selected points
    
    # Let's use a greedy approach: sort by norm2 descending, add points that don't conflict
    
    # First, let's compute the full distance matrix for the candidate set
    # n=880, so 880x880 matrix is fine
    
    dots = candidates @ candidates.T
    dist2 = norms2[:, None] + norms2[None, :] - 2 * dots
    
    # For a set S, the constraint is: min_{i,j in S, i!=j} dist2[i,j] >= max_{i in S} norms2[i]
    # This is complex. Let me simplify by requiring all selected points have the same norm2 = R,
    # and then we need dist2[i,j] >= R for all pairs.
    
    # Let's group by norm2
    unique_norms2 = np.unique(norms2)
    
    best_set = []
    best_k = 0
    
    for R in unique_norms2:
        if time.time() > deadline - 10:
            break
        mask = norms2 == R
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue
        
        # Build sub-conflict graph: edge if dist2 < R
        sub_dist2 = dist2[np.ix_(idx, idx)]
        conflict = sub_dist2 < R
        np.fill_diagonal(conflict, False)
        
        # Greedy: repeatedly pick vertex with min degree in remaining conflict graph
        remaining = set(idx.tolist())
        selected = []
        
        while remaining:
            if time.time() > deadline - 5:
                break
            # Find vertex with min conflict count in remaining
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = sum(1 for u in remaining if u != v and conflict[v, u])
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            selected.append(best_v)
            remaining.remove(best_v)
        
        if len(selected) > best_k:
            best_k = len(selected)
            best_set = selected[:]
        
        # Also try a simple greedy: pick vertices in order of increasing degree
        if time.time() > deadline - 5:
            break
        degs = np.array([sum(1 for u in idx if u != v and conflict[v, u]) for v in idx])
        order = idx[np.argsort(degs)]
        selected2 = []
        used = set()
        for v in order:
            if v in used:
                continue
            if all(not conflict[v, u] for u in selected2):
                selected2.append(v)
                used.add(v)
        
        if len(selected2) > best_k:
            best_k = len(selected2)
            best_set = selected2[:]
    
    # If we found a good set, return it
    if best_k >= 2:
        result = candidates[best_set]
        return np.around(result).astype(np.int64)
    
    # Fallback: return the original simple construction
    points = []
    for i in range(d):
        for j in range(i+1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points.append(p)
    return np.array(points)
