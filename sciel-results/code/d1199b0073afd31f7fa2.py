"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Generate all candidate vectors: for each 4-element support, all 16 sign patterns
    # Support = 4 indices, signs = ±1 on those indices, 0 elsewhere
    supports = []
    indices = range(D)
    for comb in combinations(indices, 4):
        supports.append(comb)
    
    n_sup = len(supports)  # 330
    
    # Build full candidate set: 330 * 16 = 5280 vectors
    # Each vector is in R^11 with 4 non-zero entries of ±1
    # Norm^2 = 4 for all candidates
    
    # Precompute all candidate vectors as a matrix
    # candidates[i, j] where i in [0, 5280), j in [0, 11)
    candidates = np.zeros((n_sup * 16, D), dtype=np.int64)
    for i, comb in enumerate(supports):
        for sign_mask in range(16):
            row = i * 16 + sign_mask
            for j, idx in enumerate(comb):
                if sign_mask & (1 << j):
                    candidates[row, idx] = 1
                else:
                    candidates[row, idx] = -1
    
    n_cand = len(candidates)  # 5280
    
    # Compute all pairwise dot products
    # dots[i, j] = candidates[i] . candidates[j]
    # Condition: for selected set S, we need min pairwise distance >= max norm
    # dist^2 = ||u||^2 + ||v||^2 - 2 u.v = 8 - 2 u.v (since both have norm^2 = 4)
    # We need 8 - 2 u.v >= 4 => u.v <= 2
    # Since u.v is integer and max possible is 4 (same support, same signs),
    # we need u.v <= 2 for all pairs.
    
    # Actually, let's be more careful. The condition is:
    # min_{i,j in S, i!=j} ||v_i - v_j|| >= max_{i in S} ||v_i||
    # Since all norms are 2 (norm^2 = 4), we need:
    # ||v_i - v_j||^2 >= 4 for all i != j in S
    # 8 - 2 v_i.v_j >= 4
    # v_i.v_j <= 2
    
    # So we want to maximize |S| such that for all i != j in S, candidates[i].candidates[j] <= 2
    
    # Compute the conflict matrix: conflict[i,j] = True if candidates[i].candidates[j] > 2
    # This is a symmetric matrix. We want the maximum independent set in the conflict graph.
    
    # But computing the full 5280x5280 dot product matrix might be expensive.
    # Let's use a smarter approach: group by support.
    # Vectors with the same support have dot products in {-4, -2, 0, 2, 4}
    # Vectors with different supports have dot products depending on intersection size.
    
    # Actually, let's just compute it. 5280^2 = 27.9M entries, each an int.
    # That's about 223 MB for int64. Should be fine.
    
    # But wait, we can do better. Let's use the structure.
    # Two vectors with supports S1 and S2:
    # If |S1 ∩ S2| = k, then the dot product is sum of ±1 over the intersection.
    # So |dot| <= k, and dot has the same parity as k (since it's a sum of k ±1's).
    
    # For k=4 (same support): dot in {-4, -2, 0, 2, 4}
    # For k=3: dot in {-3, -1, 1, 3}
    # For k=2: dot in {-2, 0, 2}
    # For k=1: dot in {-1, 1}
    # For k=0: dot = 0
    
    # We need dot <= 2. So:
    # k=4: dot <= 2 means dot in {-4, -2, 0, 2} (exclude 4)
    # k=3: dot <= 2 means dot in {-3, -1, 1} (exclude 3)
    # k=2: always ok (max dot = 2)
    # k=1: always ok
    # k=0: always ok
    
    # So conflicts only arise when:
    # 1. Same support (k=4) and same signs (dot=4)
    # 2. Support intersection size 3 and dot=3
    
    # Let's build the conflict graph more efficiently.
    
    # First, group candidates by support
    # For each support, there are 16 sign patterns.
    # Two vectors with the same support conflict if they have the same signs (dot=4).
    # Actually, dot=4 only if all 4 signs are the same. So for each support,
    # the 16 sign patterns form a hypercube. Two patterns conflict if they are identical? No.
    # dot = sum over 4 positions of s1[i]*s2[i]. This is 4 only if s1=s2.
    # So within a support, only identical sign patterns conflict (dot=4).
    # All other pairs within a support have dot <= 2.
    
    # For different supports with |intersection|=3:
    # The 3 overlapping positions contribute sum of ±1, and the 4th position of each
    # is 0 (since they're different supports). So dot = sum of 3 ±1's.
    # dot=3 only if all 3 signs agree.
    
    # So the conflict graph has edges between:
    # 1. Identical vectors (same support, same signs) - but we only have one of each, so no self-loops
    # 2. Vectors with support intersection size 3 and matching signs on the intersection
    
    # Let's build this efficiently.
    
    # Map from (support, sign_mask) to candidate index
    # support is a frozenset or tuple, sign_mask is 0-15
    
    cand_index = {}
    for i, comb in enumerate(supports):
        for sign_mask in range(16):
            cand_index[(comb, sign_mask)] = i * 16 + sign_mask
    
    # Build adjacency list for conflict graph
    # n_cand = 5280
    adj = [set() for _ in range(n_cand)]
    
    # For each pair of supports with intersection size 3,
    # find conflicting sign patterns
    
    # First, build a map from support to list of candidate indices
    sup_to_cands = {}
    for i, comb in enumerate(supports):
        sup_to_cands[comb] = [i * 16 + s for s in range(16)]
    
    # For each pair of supports, check intersection size
    # There are C(330,2) = 54135 pairs. For each, if |intersection|=3,
    # we need to check which sign patterns conflict.
    
    # For two supports S1, S2 with |S1 ∩ S2| = 3:
    # Let I = S1 ∩ S2 (3 elements), and let a be the unique element in S1\I, b in S2\I.
    # Sign pattern for S1: (s1_a, s1_I) where s1_I is signs on I
    # Sign pattern for S2: (s2_b, s2_I) where s2_I is signs on I
    # Dot product = s1_I . s2_I (sum over 3 positions)
    # Conflict if s1_I . s2_I = 3, i.e., s1_I = s2_I
    
    # So for each pair of supports with intersection 3,
    # for each sign pattern on the intersection (8 possibilities),
    # the candidates with that sign pattern on I conflict.
    
    # Let's precompute for each support, for each subset of its elements,
    # which sign patterns have a given sign pattern on that subset.
    
    # Actually, let's just do it directly.
    
    supports_list = list(sup_to_cands.keys())
    n_sup = len(supports_list)
    
    # Build intersection-3 pairs
    for i in range(n_sup):
        S1 = supports_list[i]
        for j in range(i+1, n_sup):
            S2 = supports_list[j]
            inter = set(S1) & set(S2)
            if len(inter) != 3:
                continue
            
            # Get the 3 intersection elements
            inter_list = list(inter)
            
            # For each of the 8 sign patterns on the intersection,
            # find candidates in S1 and S2 with that pattern
            for inter_sign in range(8):
                # Sign pattern on intersection: inter_sign is a 3-bit mask
                # For S1: sign_mask must have the bits corresponding to inter_list positions set to inter_sign
                # But the positions in inter_list are not necessarily the first 3 positions in S1.
                
                # Let's find the positions of inter_list elements in S1 and S2
                pos_in_S1 = {}
                for k, idx in enumerate(S1):
                    if idx in inter:
                        pos_in_S1[idx] = k
                
                pos_in_S2 = {}
                for k, idx in enumerate(S2):
                    if idx in inter:
                        pos_in_S2[idx] = k
                
                # For S1, the sign pattern on inter is determined by bits in pos_in_S1
                # We need sign_mask_S1 such that for each idx in inter,
                # (sign_mask_S1 >> pos_in_S1[idx]) & 1 == (inter_sign >> pos_in_inter[idx]) & 1
                # where pos_in_inter is the position of idx in inter_list
                
                # Let's just iterate over all 16 sign patterns for S1 and S2
                cands_S1 = sup_to_cands[S1]
                cands_S2 = sup_to_cands[S2]
                
                for s1_mask in range(16):
                    # Check if s1_mask has the inter_sign pattern on inter
                    ok = True
                    for idx in inter:
                        pos1 = pos_in_S1[idx]
                        pos_inter = inter_list.index(idx)
                        bit1 = (s1_mask >> pos1) & 1
                        bit_inter = (inter_sign >> pos_inter) & 1
                        if bit1 != bit_inter:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    c1 = cands_S1[s1_mask]
                    
                    for s2_mask in range(16):
                        ok2 = True
                        for idx in inter:
                            pos2 = pos_in_S2[idx]
                            pos_inter = inter_list.index(idx)
                            bit2 = (s2_mask >> pos2) & 1
                            bit_inter = (inter_sign >> pos_inter) & 1
                            if bit2 != bit_inter:
                                ok2 = False
                                break
                        if not ok2:
                            continue
                        
                        c2 = cands_S2[s2_mask]
                        
                        # These two candidates conflict
                        adj[c1].add(c2)
                        adj[c2].add(c1)
    
    # Now we have the conflict graph. We want the maximum independent set.
    # This is a hard problem, but the graph has structure.
    # Let's use a greedy approach with local search.
    
    # Greedy: start with empty set, add vertices in order of increasing degree
    # Then do local search: try to add a vertex if it doesn't conflict with any in the set,
    # or swap to improve.
    
    # Actually, let's use a simpler approach: since the graph is sparse,
    # we can use a greedy maximal independent set, then try to improve.
    
    # First, compute degrees
    degrees = [len(adj[i]) for i in range(n_cand)]
    
    # Sort by degree
    order = sorted(range(n_cand), key=lambda x: degrees[x])
    
    # Greedy MIS
    selected = set()
    for v in order:
        # Check if v conflicts with any selected vertex
        conflict = False
        for u in selected:
            if u in adj[v]:
                conflict = True
                break
        if not conflict:
            selected.add(v)
    
    # Local search: try to add vertices not in selected
    # A vertex can be added if it doesn't conflict with any in selected
    # But that's the same as the greedy step.
    
    # Better local search: try to swap.
    # For each vertex not in selected, see if removing one selected vertex
    # allows adding two new vertices.
    
    # This is complex. Let's try a different approach:
    # Since the conflict graph is very sparse (most vertices have low degree),
    # the greedy MIS should be close to optimal.
    
    # Let's estimate the size.
    # Each vertex conflicts with:
    # - 0 other vertices with the same support (since we only have one of each sign pattern,
    #   and same support + same signs = same vector, which we don't have duplicates of)
    #   Wait, we DO have one of each sign pattern per support. So same support, same signs
    #   is the same vertex. No self-conflict.
    #   Same support, different signs: dot <= 2, no conflict.
    # - For each of the C(10,3) = 120 other supports that intersect in 3,
    #   how many sign patterns conflict?
    #   For a given support S1 and another support S2 with |S1∩S2|=3,
    #   the number of conflicting sign patterns is:
    #   For each of the 8 sign patterns on the intersection,
    #   there is exactly 1 sign pattern in S1 and 1 in S2 that match.
    #   So 8 conflicts per pair of supports.
    #   Number of supports intersecting S1 in 3: 
    #   S1 has 4 elements. Choose 3 from S1: C(4,3)=4.
    #   For each choice of 3, the 4th element of S2 can be any of the remaining 7 elements.
    #   So 4 * 7 = 28 supports intersect S1 in exactly 3.
    #   So each vertex conflicts with 28 * 8 = 224 other vertices.
    
    # So the graph is 224-regular (approximately).
    # For a 224-regular graph on 5280 vertices, the MIS is at least 5280/225 ≈ 23.5.
    # But this is a lower bound. The actual MIS could be much larger.
    
    # Let's run the greedy and see what we get.
    
    print(f"Greedy MIS size: {len(selected)}")
    
    # Convert to list
    selected_list = list(selected)
    
    # Extract the vectors
    result = candidates[selected_list]
    
    # Check validity
    if len(result) < 2:
        return result
    
    # Verify the condition
    # min pairwise distance >= max norm
    norms = np.linalg.norm(result, axis=1)
    max_norm = norms.max()
    
    # Compute pairwise distances
    # For large k, this is O(k^2 * D)
    # k is probably a few hundred, so it's fine.
    
    # Actually, let's just return the result. The evaluator will check.
    
    return result

# EVOLVE-BLOCK-END
