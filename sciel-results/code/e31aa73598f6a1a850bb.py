"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _generate_supports():
    """Generate all 4-element subsets of {0, ..., D-1}."""
    supports = list(combinations(range(D), 4))
    return supports

def _generate_signs_for_support(support):
    """Generate all 16 sign vectors for a given 4-element support."""
    signs = []
    for mask in range(16):
        v = np.zeros(D, dtype=np.int64)
        for j, idx in enumerate(support):
            if mask & (1 << j):
                v[idx] = 1
            else:
                v[idx] = -1
        signs.append(v)
    return np.array(signs, dtype=np.int64)

def construct():
    deadline = time.time() + 110
    
    # 1. Generate all possible 4-element supports
    supports = _generate_supports()
    n_supports = len(supports) # 330
    
    # 2. Precompute compatibility between supports
    # Two supports are compatible if their intersection size is <= 2.
    # If intersection is 3 or 4, they are incompatible (cannot both be in the final set of supports 
    # if we want to maximize distinct supports? No, the constraint is on vectors.
    # Actually, if two supports share 3 indices, any vector from support A and any vector from support B 
    # will have an inner product of at least 1*1 + 1*1 + 1*1 = 3 (if signs align) or -3.
    # Wait, the condition is |<u,v>| <= M^2 - 1 = 3.
    # If supports share 3 indices, the inner product is sum of 3 products of +/-1.
    # The max possible inner product is 3. So |<u,v>| can be 3.
    # The condition is min pairwise distance >= max norm.
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2<u,v> = 4 + 4 - 2<u,v> = 8 - 2<u,v>.
    # We need 8 - 2<u,v> >= 4 => 4 >= 2<u,v> => <u,v> <= 2.
    # AND 8 - 2<u,v> >= 4 => -4 <= -2<u,v> => <u,v> >= -2.
    # So we require -2 <= <u,v> <= 2.
    
    # If supports share 3 indices, <u,v> is sum of 3 terms +/-1. Values: -3, -1, 1, 3.
    # So if supports share 3 indices, we MUST ensure signs are chosen such that <u,v> is -1 or 1.
    # This is a constraint on the signs.
    
    # If supports share 4 indices (same support), <u,v> is sum of 4 terms. Values: -4, -2, 0, 2, 4.
    # We need -2 <= <u,v> <= 2. So we cannot have <u,v> = 4 (identical signs) or -4 (opposite signs).
    # So for the same support, we can pick at most 8 out of 16 signs? 
    # Specifically, we need to pick a set of signs from the 16 such that pairwise dot products are in {-2, 0, 2}.
    # This is equivalent to finding a max code in the hypercube Q4 with min distance 4?
    # Distance squared = 4 + 4 - 2ip = 8 - 2ip.
    # If ip=2, dist^2=4. If ip=0, dist^2=8. If ip=-2, dist^2=12.
    # We need dist^2 >= 4. So ip <= 2 is the binding constraint for same support?
    # Wait, if ip=2, dist^2=4. If ip=-2, dist^2=12. Both are >= 4.
    # So for same support, we just need to avoid ip=4 and ip=-4.
    # ip=4 means u=v. ip=-4 means u=-v.
    # So we can pick at most one from each pair {u, -u}.
    # There are 8 such pairs. So max 8 vectors per support.
    
    # If supports share 2 indices, <u,v> is sum of 2 terms. Values: -2, 0, 2.
    # All are in [-2, 2]. So ANY signs are compatible if supports share <= 2 indices.
    
    # If supports share 1 index, <u,v> is +/-1. Always compatible.
    
    # If supports share 0 indices, <u,v> = 0. Always compatible.
    
    # So the only constraints are:
    # 1. Within the same support: cannot pick both u and -u. (Max 8 vectors per support).
    # 2. Between supports sharing 3 indices: must pick signs such that <u,v> in {-1, 1}.
    
    # Strategy:
    # 1. Select a subset of supports S such that no two supports share 3 indices?
    #    If we select supports that share 3 indices, we have complex sign constraints.
    #    If we select supports that share <= 2 indices, there are NO sign constraints between different supports.
    #    This simplifies the problem significantly!
    #    Let's find the maximum number of 4-subsets of {0..10} such that any two intersect in at most 2 elements.
    #    This is a packing problem.
    
    # Let's estimate the size.
    # Total supports = 330.
    # Conflict if intersection >= 3.
    # We want a Max Independent Set on the graph where edges connect supports with intersection >= 3.
    
    # Let's build this graph.
    # n=330. 
    # Intersection size calculation:
    # For each pair, check intersection.
    
    adj = np.zeros((n_supports, n_supports), dtype=bool)
    for i in range(n_supports):
        for j in range(i + 1, n_supports):
            # Check intersection size
            # supports are tuples of 4 ints
            inter = set(supports[i]).intersection(supports[j])
            if len(inter) >= 3:
                adj[i, j] = True
                adj[j, i] = True
                
    # Find Max Independent Set on this graph.
    # This graph is likely sparse enough for a greedy + local search approach to work well.
    # Or we can use a simple greedy with restarts.
    
    best_supports = []
    best_k = 0
    
    # Greedy MIS with restarts
    for restart in range(50):
        if time.time() > deadline - 20:
            break
            
        # Shuffle nodes
        order = np.random.permutation(n_supports)
        
        selected = []
        selected_set = set()
        
        for node in order:
            # Check if node conflicts with any selected node
            # We can maintain a list of selected nodes and check against them
            # Or maintain a bitmask of conflicts?
            # Since n=330, checking against selected list is O(K).
            # Total O(N*K). N=330, K~50. 16500 ops per restart. Very fast.
            
            conflict = False
            for s in selected:
                if adj[node, s]:
                    conflict = True
                    break
            
            if not conflict:
                selected.append(node)
                selected_set.add(node)
                
        if len(selected) > best_k:
            best_k = len(selected)
            best_supports = selected.copy()
            
    # We have best_supports.
    # For each support in best_supports, we can pick up to 8 vectors (one from each +/- pair).
    # Since there are no sign constraints between different supports (intersection <= 2),
    # we can simply pick 8 vectors for each support.
    # Which 8? It doesn't matter for the count, as long as we pick 8 distinct ones that are not negatives of each other.
    # Let's just pick the first 8 from the 16, ensuring no u, -u pairs.
    # Actually, just picking any 8 from the 8 pairs is fine.
    # Let's pick the 8 vectors with first bit 0? 
    # Indices 0..15.
    # Pair (i, i^8)? No, u and -u correspond to mask and mask^15.
    # So we can pick masks 0, 1, 2, 3, 4, 5, 6, 7.
    # These are 8 vectors. None are negatives of each other?
    # mask 0: 0000. mask 15: 1111. -v0 = v15.
    # mask 1: 0001. mask 14: 1110. -v1 = v14.
    # So picking 0..7 ensures we don't pick both u and -u.
    
    all_vectors = []
    for s_idx in best_supports:
        support = supports[s_idx]
        # Generate 8 vectors
        for mask in range(8):
            v = np.zeros(D, dtype=np.int64)
            for j, idx in enumerate(support):
                if mask & (1 << j):
                    v[idx] = 1
                else:
                    v[idx] = -1
            all_vectors.append(v)
            
    result = np.array(all_vectors, dtype=np.int64)
    
    # Optional: Can we add more vectors?
    # We picked 8 per support. Can we pick more?
    # No, max 8 per support because of the u, -u constraint.
    # Can we add vectors from supports NOT in best_supports?
    # If a support s' is not in best_supports, it conflicts with some s in best_supports (intersection >= 3).
    # If we add a vector v' from s', it must satisfy |<v', v>| <= 2 for all v in result.
    # Since s' and s share 3 indices, <v', v> depends on signs.
    # We might be able to add SOME vectors from conflicting supports if we choose signs carefully.
    # But for simplicity and speed, let's stick to the clean packing.
    # The score is likely dominated by the number of supports * 8.
    
    # Let's verify the result validity quickly.
    # If result is empty, fallback.
    if len(result) == 0:
         # Fallback to simple shell 2
         vecs = []
         for i in range(D):
             for j in range(i+1, D):
                 for si in (1, -1):
                     for sj in (1, -1):
                         v = np.zeros(D, dtype=np.int64)
                         v[i] = si
                         v[j] = sj
                         vecs.append(v)
         result = np.array(vecs, dtype=np.int64)
         
    return result

# EVOLVE-BLOCK-END
