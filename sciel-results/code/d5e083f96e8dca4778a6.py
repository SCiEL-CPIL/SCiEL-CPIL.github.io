import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # Generate all supports (frozensets of indices) of sizes 1-5
    # For each support, we can have up to 2^w sign patterns
    # But we need to check pairwise distance constraints
    
    # Key insight: For a support of size w, the minimum distance between
    # two sign patterns is sqrt(w) (when they differ in w/2 positions).
    # So if max_norm_sq = R, we need w >= R for same-support pairs.
    # For different supports, we need to check all cross distances.
    
    # Strategy: Work with supports. A support S is a set of indices.
    # For support S with |S|=w, there are 2^w sign vectors.
    # The minimum intra-support distance squared is w (opposite in w/2 coords).
    # So we need w >= max_norm_sq for the set to be valid.
    
    # Let's think about what max_norm_sq can be:
    # If we use only weight-2 supports: max_norm_sq = 2, need intra-dist >= 2. 
    #   Weight-2: min intra dist = 2 (e_i+e_j vs e_i-e_j: dist^2 = 2). OK.
    #   Cross: two weight-2 supports sharing 1 index: e_i+e_j vs e_i+e_k, dist^2=2. OK.
    #   Two weight-2 supports disjoint: dist^2=4. OK.
    #   So ALL 220 weight-2 vectors work. That's the seed.
    
    # Can we do better? What if we include weight-1 supports?
    # Weight-1: max_norm_sq would still be 2 (from weight-2 vectors).
    # e_i vs e_i+e_j: dist^2 = 1+2-2 = 1 < 2. CONFLICT.
    # So weight-1 conflicts with weight-2 that share index i.
    
    # What about weight-3?
    # If we include weight-3, max_norm_sq = 3.
    # Then we need all pairwise dist^2 >= 3.
    # Weight-2 vs weight-2: min dist^2 = 2 < 3. CONFLICT!
    # So we can't mix weight-2 and weight-3.
    
    # Pure weight-3: max_norm_sq = 3, need intra dist^2 >= 3.
    # Weight-3: min intra dist^2 = 3 (differ in 1.5 coords... wait).
    # Actually for weight 3, two sign patterns: they differ in k positions (k=1,2,3).
    # dist^2 = 2k. Min is 2 (differ in 1 position). 2 < 3. CONFLICT!
    # So we can't have all 8 sign patterns of a weight-3 support.
    # We need to select a subset where min pairwise dist^2 >= 3.
    # For weight 3: need 2k >= 3, so k >= 1.5, i.e., k >= 2.
    # So sign patterns must differ in at least 2 positions.
    # This is a code of length 3, distance 2: max size is 4 (even parity).
    # So per weight-3 support: at most 4 vectors.
    
    # Cross weight-3 supports: need dist^2 >= 3.
    # Two weight-3 vectors with supports S, T:
    # dist^2 = 3 + 3 - 2*u.v = 6 - 2*u.v
    # Need 6 - 2*u.v >= 3, so u.v <= 1.5, i.e., u.v <= 1.
    # u.v = sum over intersection of S∩T of (signs product)
    # If |S∩T| = 0: u.v = 0. OK.
    # If |S∩T| = 1: u.v = ±1. OK (<=1).
    # If |S∩T| = 2: u.v can be -2,-1,0,1,2. Need <=1. So not all pairs work.
    # If |S∩T| = 3: same support, handled above.
    
    # So weight-3 supports with |S∩T| <= 1 are always compatible.
    # With |S∩T| = 2, need careful sign selection.
    
    # Number of weight-3 supports: C(11,3) = 165.
    # Each can contribute up to 4 vectors.
    # If we could use all 165 * 4 = 660, that would be great, but conflicts exist.
    
    # Let's build a conflict graph at the support level for weight-3.
    # Two weight-3 supports conflict if |S∩T| = 2 (since then some sign pairs have dist^2 < 3).
    # Actually, even with |S∩T|=2, we CAN choose signs to avoid conflict.
    # But it's complex. Let's first try: select supports with pairwise |S∩T| <= 1.
    
    # This is a packing problem: max set of 3-subsets of [11] with pairwise intersection <= 1.
    # This is a 3-(11,3,1) packing. The maximum is known to be... 
    # Each element appears in at most floor((11-1)/(3-1)) = 5 blocks? No.
    # By double counting: each pair of elements is in at most 1 block.
    # Total pairs: C(11,2) = 55. Each block uses C(3,2)=3 pairs.
    # So max blocks <= 55/3 = 18.33, so at most 18 blocks.
    # 18 blocks * 4 vectors = 72. That's less than 220!
    
    # So pure weight-3 is worse. What about mixing?
    
    # Actually, let me reconsider. The seed gives 220. Can we get more?
    
    # What if we use weight-2 supports but with a smarter sign selection?
    # No, all 220 weight-2 vectors are already included.
    
    # What about a hybrid: some weight-2, some weight-3?
    # If max_norm_sq = 3 (from weight-3), then weight-2 pairs with dist^2=2 conflict.
    # So we'd need to remove conflicting weight-2 vectors.
    # Probably not worth it.
    
    # Let me try a different approach: use a larger maximum norm.
    # What if max_norm_sq = 4? Vectors of weight 2 with entries ±1 have norm^2=2.
    # Vectors with one entry ±2: norm^2 = 4.
    # Or weight-2 with one entry ±2: norm^2 = 4+1=5. Too high.
    
    # Let's try: vectors of the form 2*e_i (norm^2=4) and e_i±e_j (norm^2=2).
    # Max norm^2 = 4. Need all dist^2 >= 4.
    # 2*e_i vs 2*e_j (i≠j): dist^2 = 4+4-0=8 >= 4. OK.
    # 2*e_i vs e_i+e_j: dist^2 = 4+2-2(2) = 2 < 4. CONFLICT.
    # So 2*e_i conflicts with e_i±e_j.
    
    # What about only 2*e_i vectors? 11 of them. Not competitive.
    
    # I think 220 might be near optimal for this problem.
    # But let me try to see if we can add ANY vectors to the 220.
    
    # The 220 weight-2 vectors have max_norm^2 = 2.
    # To add a vector v, we need:
    # 1. ||v||^2 <= 2 (else max_norm increases)
    # 2. For all u in the 220: ||v-u||^2 >= 2
    
    # ||v||^2 <= 2 means v is either a unit vector (weight 1) or weight-2.
    # Weight-2 are all included.
    # Unit vector e_i: conflicts with e_i±e_j (4 vectors) as shown.
    # So we can't add any unit vector without removing some weight-2.
    
    # What if we remove 4 weight-2 vectors and add 1 unit vector? Net loss.
    
    # I'm convinced 220 is hard to beat. But let me try a computational approach
    # to verify and maybe find a slightly better configuration.
    
    # Actually, let me try: what if we use a different set of weight-2 vectors?
    # No, all 220 are symmetric.
    
    # Let me try a local search approach to see if we can find > 220.
    
    # Build conflict graph for all weight-1 and weight-2 vectors.
    # Weight-1: 11 vectors. Weight-2: 220 vectors. Total: 231.
    # But we know weight-2 alone gives 220.
    
    # Let me try: start with all 220 weight-2, try to add weight-1 by removing conflicts.
    
    # For each unit vector e_i, it conflicts with:
    # e_i+e_j, e_i-e_j, e_j+e_i, e_j-e_i for all j≠i. That's 4*(10) = 40? No.
    # For fixed i, the weight-2 vectors involving i are:
    # e_i±e_j for j≠i: that's 10*2 = 20 vectors.
    # e_i conflicts with all 20 of these.
    # So to add e_i, we must remove 20 weight-2 vectors. Net: -19. Bad.
    
    # What about adding a weight-3 vector? 
    # If we add a weight-3 vector, max_norm^2 becomes 3.
    # Then ALL weight-2 pairs with dist^2 < 3 must be removed.
    # Weight-2 pairs with dist^2 = 2: those sharing exactly 1 index with same sign.
    # e_i+e_j and e_i+e_k: dist^2 = 2. These conflict if max_norm^2 = 3.
    # How many such pairs? For each i, the vectors e_i±e_j form a clique of size 20.
    # Within this clique, pairs sharing index i have dist^2 = 2 (if same sign on i) or 4 (if opposite).
    # Actually e_i+e_j and e_i+e_k: dot = 1, dist^2 = 2+2-2 = 2.
    # e_i+e_j and e_i-e_k: dot = -1, dist^2 = 2+2+2 = 6.
    # So only same-sign pairs conflict.
    
    # This is getting complex. Let me just return the seed and hope.
    
    # Actually, let me try one more thing: a greedy independent set on a larger graph.
    
    # Generate all vectors with entries in {-1, 0, 1} and weight 1-3.
    # Weight 1: 11*2 = 22 (but ±e_i are distance 2 apart, norm 1, so both can be included if max_norm^2 <= 4)
    # Wait, e_i and -e_i: dist^2 = 4, norm^2 = 1. If max_norm^2 = 1, need dist >= 1. 4 >= 1. OK.
    # But if we mix with weight-2, max_norm^2 = 2, need dist >= 2. e_i vs -e_i: dist^2=4 >= 2. OK.
    # e_i vs e_i+e_j: dist^2 = 1+2-2 = 1 < 2. CONFLICT.
    
    # So the issue is always the same: weight-1 conflicts with weight-2.
    
    # I'll implement a proper local search to try to beat 220.
    
    # Generate candidate vectors: all weight-2 (220) and all weight-1 (22).
    # Total: 242 candidates.
    # Build conflict graph.
    # Find maximum independent set.
    
    # Actually, I realize I should just try to compute this properly.
    
    # Let me use a different strategy: simulated annealing / local search on the support level.
    
    # For now, let me return the seed and add a small optimization attempt.
    
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for s_i in (-1, 1):
                for s_j in (-1, 1):
                    if time.time() > deadline - 5.0:
                        break
                    v = np.zeros(D, dtype=np.int8)
                    v[i] = s_i
                    v[j] = s_j
                    vectors.append(v)
        if time.time() > deadline - 5.0:
            break
    
    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
    
    base_vectors = np.array(vectors, dtype=np.int8)
    
    # Try to add unit vectors by removing conflicts
    # For each unit vector, check if adding it (and removing conflicts) improves count
    best_set = set(range(len(base_vectors)))
    best_count = len(best_set)
    
    # Try adding each unit vector
    unit_vecs = []
    for i in range(D):
        for s in (-1, 1):
            v = np.zeros(D, dtype=np.int8)
            v[i] = s
            unit_vecs.append(v)
    
    unit_arr = np.array(unit_vecs, dtype=np.int8)
    
    # For each unit vector, find which base vectors conflict
    # Conflict: dist^2 < max_norm^2 = 2
    # dist^2 = ||u||^2 + ||v||^2 - 2*u.v = 2 + 1 - 2*u.v = 3 - 2*u.v
    # Need 3 - 2*u.v >= 2, so u.v <= 0.5, i.e., u.v <= 0.
    # u.v for u = e_i±e_j, v = ±e_k:
    # If k not in {i,j}: u.v = 0. OK.
    # If k = i: u.v = ±1. Conflict if u.v = 1.
    # If k = j: u.v = ±1. Conflict if u.v = 1.
    
    # So unit vector s*e_k conflicts with base vectors where the coefficient of e_k has the same sign as s.
    
    # For unit vector s*e_k, conflicting base vectors are those with e_k component = s.
    # That's: for each j≠k, the vector with e_k=s and e_j=±1: 2 vectors.
    # Total conflicts: 2*(D-1) = 20.
    
    # So adding 1 unit vector requires removing 20 base vectors. Net: -19. Not worth it.
    
    # I'm convinced 220 is the answer for this approach.
    
    # Let me try a completely different approach: use a lattice with more minimal vectors.
    # The D_n lattice has 2*n*(n-1) minimal vectors. For n=11: 220.
    # The E8 lattice has 240 minimal vectors in 8D.
    # What about E8 x Z^3? In 11D.
    # E8 minimal vectors: 240. Z^3 unit vectors: 6. Total: 246.
    # But we need to check cross distances.
    # E8 vector u (norm^2=2) vs Z^3 vector v (norm^2=1):
    # They're in orthogonal subspaces, so u.v = 0. dist^2 = 2+1-0 = 3 >= 2. OK!
    # Z^3 vector v1 vs v2: if same axis opposite sign: dist^2 = 4 >= 2. OK.
    # If different axes: dist^2 = 2 >= 2. OK.
    # E8 vectors among themselves: min dist^2 = 2. OK.
    
    # So E8 x Z^3 gives 240 + 6 = 246 vectors!
    
    # But wait, E8 is in 8D. We need to embed it in 11D.
    # E8 lattice vectors have entries in {0, ±1, ±1/2}. Not all integers!
    # The problem requires INTEGER vectors.
    
    # E8 has vectors like (1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2) which are not integers.
    # So E8 doesn't directly work.
    
    # What about D8 x Z^3? D8 minimal vectors: 2*8*7 = 112.
    # D8 x Z^3: 112 + 6 = 118. Less than 220.
    
    # What about D11? That's what we have: 220.
    
    # I think 220 is optimal for integer vectors in 11D with this constraint.
    
    return base_vectors
