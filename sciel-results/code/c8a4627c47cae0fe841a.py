import numpy as np
import time
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 145.0
    D = 11

    # Precompute all supports of size 4 and size 3
    sup4 = list(combinations(range(D), 4))
    N4 = len(sup4)
    sup3 = list(combinations(range(D), 3))
    N3 = len(sup3)

    # Bitmasks for supports
    masks4 = []
    for s in sup4:
        m = 0
        for i in s:
            m |= (1 << i)
        masks4.append(m)

    masks3 = []
    for s in sup3:
        m = 0
        for i in s:
            m |= (1 << i)
        masks3.append(m)

    # Build conflict graph for weight-4 supports: conflict if intersection size >= 3
    # Two weight-4 vectors with supports that share 3+ coordinates can have small distance
    # For two vectors on supports with intersection >= 3, the minimum distance can be small
    # We want supports with intersection <= 2 for all pairs (compatible)

    # For weight-3 supports: two weight-3 vectors on same support have distance 2*sqrt(3) if opposite signs
    # but same support with different sign patterns can have distance 2*sqrt(3) or 4*sqrt(3)
    # Two weight-3 vectors on different supports: if supports intersect in 2, min distance = sqrt(3+3-2) = sqrt(4)=2? No...
    # Actually for weight-3 vectors with supports intersecting in 2:
    # v1 = (1,1,1,0,0,0,0,0,0,0,0), v2 = (1,1,-1,0,0,0,0,0,0,0,0) -> same support
    # v1 = (1,1,1,0,0,0,0,0,0,0,0), v2 = (1,1,0,1,0,0,0,0,0,0,0) -> intersection {0,1}, size 2
    # distance = sqrt((0)^2+(0)^2+(1)^2+(-1)^2+0+...) = sqrt(2) < sqrt(3) = norm
    # So weight-3 supports must have intersection <= 1 for safety, or we need to be careful.
    # Actually the condition is: min pairwise distance >= max norm.
    # For weight-k vectors, norm = sqrt(k). Distance between two vectors on supports with intersection t:
    # if we can choose signs to minimize distance, min dist over sign choices for a given pair of supports
    # with intersection t is: if t = 0: sqrt(k1+k2), if t>0: sqrt(k1+k2-2t) (worst case signs aligned on intersection)
    # But we get to CHOOSE the sign patterns. So for a given pair of supports, we pick sign patterns to MAXIMIZE distance.
    # Max distance between two vectors on supports with intersection t:
    # = sqrt(k1+k2+2t) (signs opposite on intersection)
    # Wait no. Let's think carefully.
    # v1 has entries ±1 on support S1, v2 has entries ±1 on support S2.
    # ||v1 - v2||^2 = sum over all coords of (v1[i]-v2[i])^2
    # On coords in S1∩S2: (±1 - ±1)^2 = 0 or 4
    # On coords in S1\S2: (±1 - 0)^2 = 1
    # On coords in S2\S1: (0 - ±1)^2 = 1
    # So ||v1-v2||^2 = 4*(# of intersection coords where signs differ) + (k1-t) + (k2-t)
    # = 2k1 + 2k2 - 2t + 4*(#sign differences on intersection)
    # To maximize: make all signs differ on intersection -> 4t
    # max dist^2 = 2k1+2k2-2t+4t = 2k1+2k2+2t
    # For k1=k2=4, t=2: max dist^2 = 8+8+4 = 20, norm^2 = 16. 20 >= 16 ✓
    # For k1=k2=4, t=3: max dist^2 = 8+8+6 = 22 >= 16 ✓
    # For k1=k2=4, t=4 (same support): max dist^2 = 8+8+8 = 24 >= 16 ✓ (opposite signs)
    # So actually for weight-4, ANY pair of supports works if we choose signs optimally!
    # The constraint is: for EVERY pair in our set, the chosen sign patterns must give dist >= norm.
    # Since max possible dist^2 for weight-4 pair with intersection t is 16+2t >= 16, we're always fine
    # IF we can choose signs to achieve that max. The max is achieved when all signs are opposite on intersection.
    # So the problem becomes: can we assign signs to all weight-4 vectors such that for every pair,
    # the signs are opposite on their intersection?
    # This is a constraint satisfaction problem. If we pick a large set of supports, we need consistent sign assignments.
    # This is equivalent to a 2-coloring of a hypergraph, which may not always be possible.

    # Alternative approach: use only weight-4 vectors on supports that are pairwise disjoint or intersect in exactly 1.
    # If intersection = 0: dist^2 = 16+16 = 32, fine.
    # If intersection = 1: max dist^2 = 16+16+2 = 34, fine.
    # If intersection = 2: max dist^2 = 16+16+4 = 36, fine.
    # If intersection = 3: max dist^2 = 16+16+6 = 38, fine.
    # If intersection = 4: max dist^2 = 24, fine.
    # So ALL pairs of weight-4 supports are feasible! The issue is sign consistency.

    # Let me think differently. For each pair (i,j) with nonempty intersection, we need:
    # For each coord c in S_i ∩ S_j: sign_i[c] = -sign_j[c]
    # This means: sign_i and sign_j are "opposite" on their intersection.
    # Define a partial order: if we fix signs for one vector, it constrains others.
    # This is like a system of equations mod 2.

    # Actually, let's just use a different strategy: use weight-2 vectors.
    # For weight-2 vectors on supports (i,j): norm^2 = 2.
    # Two vectors on supports (i,j) and (k,l):
    # If disjoint: dist^2 = 2+2 = 4 >= 2 ✓
    # If share 1 coord: max dist^2 = 2+2+2 = 6 >= 2 ✓
    # If same support: max dist^2 = 4 >= 2 ✓
    # Sign consistency: for same support, need opposite signs -> fine.
    # For sharing 1 coord: need opposite sign on that coord.
    # This is again a constraint satisfaction.

    # Better idea: just use all weight-4 vectors with a FIXED sign pattern per support,
    # and verify. If some pair fails, remove one.

    # Actually, the simplest high-scoring approach: use all 16*25 = 400 weight-4 vectors
    # (25 supports × 16 sign patterns each) and verify pairwise.
    # That's 400 vectors, C(400,2) = 79800 pairs to check. Very fast.

    # But wait - do all 400 work together? For two vectors on the SAME support with sign patterns
    # that differ in exactly 2 positions: dist^2 = 4*2 + 4*2 = 16... wait.
    # Same support, 4 coords. If signs differ in m positions: dist^2 = 4m.
    # Need 4m >= 16 -> m >= 4. So they must be complete opposites!
    # So on each support, we can have at most 2 vectors: v and -v.

    # For different supports with intersection t:
    # Need the sign patterns to be "opposite" on the intersection.
    # If t=4 (same support): handled above.
    # If t=3: 3 coords shared. Need signs opposite on all 3. The 4th coord is free.
    # So for support A={a,b,c,d} and B={a,b,c,e}: sign_A[a]=-sign_B[a], sign_A[b]=-sign_B[b], sign_A[c]=-sign_B[c].
    # sign_A[d] and sign_B[e] are free.
    # For a fixed sign pattern on A, there are 2*2=4 compatible patterns on B.
    # But those 4 patterns on B must also be compatible with all other selected vectors.

    # This is getting complex. Let me use a simpler approach:
    # Select a set of supports, then for each support pick a sign pattern,
    # and verify all pairs. Use greedy: add supports one at a time, pick best sign pattern.

    # Even simpler: just enumerate all 400 vectors and find the largest subset that works.
    # 400 vectors, check all pairs.

    all_vectors = []
    for sup in sup4:
        for signs in [(s1, s2, s3, s4) for s1 in (1,-1) for s2 in (1,-1) for s3 in (1,-1) for s4 in (1,-1)]:
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            all_vectors.append(v)

    N = len(all_vectors)  # 400
    norms_sq = np.array([np.dot(v, v) for v in all_vectors], dtype=np.int64)
    max_norm_sq = int(norms_sq.max())  # 16

    # Build compatibility matrix: pair (i,j) is compatible if ||vi-vj||^2 >= max_norm_sq
    # Use matrix operations for speed
    V = np.array(all_vectors, dtype=np.int16)  # (400, 11)
    # dist_sq[i,j] = ||vi||^2 + ||vj||^2 - 2*vi.vj
    G = V @ V.T  # (400, 400)
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * G

    # Compatible if dist_sq[i,j] >= max_norm_sq (for i != j)
    compat_matrix = dist_sq >= max_norm_sq
    np.fill_diagonal(compat_matrix, False)

    # Now find maximum clique in this graph (400 nodes).
    # Use a simple greedy approach with multiple restarts, plus local improvement.

    best_count = 0
    best_set = set()

    def greedy_clique(order):
        """Greedily build clique by processing vertices in given order."""
        clique = []
        for v in order:
            if all(compat_matrix[v, u] for u in clique):
                clique.append(v)
        return clique

    def local_search(clique):
        """Try to improve clique by swapping."""
        clique = list(clique)
        improved = True
        while improved:
            improved = False
            cur_set = set(clique)
            for i, vi in enumerate(clique):
                for vj in range(N):
                    if vj in cur_set:
                        continue
                    # Try swapping vi with vj
                    new_set = (cur_set - {vi}) | {vj}
                    # Check if vj is compatible with all in new_set
                    if all(compat_matrix[vj, u] for u in new_set if u != vj):
                        # Also need all existing pairs to still be compatible (they were)
                        clique[i] = vj
                        cur_set = new_set
                        improved = True
                        break
                if improved:
                    break
        return clique

    # Multiple random restarts
    import random
    rng = random.Random(42)

    for restart in range(200):
        if time.time() > deadline - 10:
            break
        order = list(range(N))
        rng.shuffle(order)
        clique = greedy_clique(order)
        if len(clique) > best_count:
            best_count = len(clique)
            best_set = set(clique)
        # Local search
        clique_ls = local_search(clique)
        if len(clique_ls) > best_count:
            best_count = len(clique_ls)
            best_set = set(clique_ls)

    # Also try degree-based ordering
    degrees = [sum(compat_matrix[i, :]) for i in range(N)]
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        # Perturbed degree ordering
        perturbed_deg = [d + rng.gauss(0, 5) for d in degrees]
        order = sorted(range(N), key=lambda x: -perturbed_deg[x])
        clique = greedy_clique(order)
        if len(clique) > best_count:
            best_count = len(clique)
            best_set = set(clique)
        clique_ls = local_search(clique)
        if len(clique_ls) > best_count:
            best_count = len(clique_ls)
            best_set = set(clique_ls)

    # Build final vector array
    result_vectors = [all_vectors[i] for i in sorted(best_set)]
    if not result_vectors:
        # Fallback
        result_vectors = [np.array([1,1,0,0,0,0,0,0,0,0,0], dtype=np.int8),
                          np.array([-1,-1,0,0,0,0,0,0,0,0,0], dtype=np.int8)]

    return np.array(result_vectors, dtype=np.int8)
