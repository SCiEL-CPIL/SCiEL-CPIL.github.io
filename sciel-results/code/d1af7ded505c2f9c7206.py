import numpy as np
import time
from itertools import combinations, product
from collections import defaultdict

def kissing_number11() -> np.ndarray:
    d = 11
    t0 = time.time()
    deadline = t0 + 580  # Safety margin
    
    def elapsed():
        return time.time() - t0
    
    # Generate all ±1 vectors with Hamming weight 4 in 11D
    # Each vector has exactly 4 non-zero entries, each ±1
    # Count: C(11,4) * 2^4 = 330 * 16 = 5280
    # Norm squared = 4 for all such vectors
    # Two such vectors conflict if dist² < 4, i.e., |v-w|² < 4
    # |v-w|² = |v|² + |w|² - 2v·w = 8 - 2v·w
    # So conflict iff 8 - 2v·w < 4 iff v·w > 2
    # v·w = sum of products over all coordinates
    # Since v,w have ±1 in exactly 4 positions each, v·w depends on overlap
    # If they share k positions with same sign and l positions with opposite sign:
    # v·w = k - l, where k+l <= 4 (overlap size)
    # Actually, let's think differently. v·w = sum_i v_i * w_i
    # Both have 4 non-zero entries. Let S = support(v), T = support(w)
    # On S∩T: v_i * w_i = ±1
    # On S\T or T\S: one is 0, so product is 0
    # So v·w = sum over i in S∩T of (v_i * w_i)
    # Let m = |S∩T|, and let s = number of positions in S∩T where v_i = w_i
    # Then v·w = s - (m-s) = 2s - m
    # Conflict iff 2s - m > 2 iff 2s > m + 2
    # For m=0: 0 > 2? No. No conflict.
    # For m=1: 2s > 3, s can be 0 or 1. 2*1=2 > 3? No. No conflict.
    # For m=2: 2s > 4, s can be 0,1,2. 2*2=4 > 4? No. No conflict.
    # For m=3: 2s > 5, s can be 0,1,2,3. 2*3=6 > 5? Yes! Conflict if s=3.
    #   s=3 means all 3 overlapping positions have same sign.
    # For m=4: 2s > 6, s can be 0,1,2,3,4. 2*4=8 > 6? Yes! Conflict if s>=4.
    #   s=4 means all 4 positions have same sign (i.e., v=w on their support).
    #   But also s=3: 2*3=6 > 6? No. So only s=4 conflicts for m=4.
    #   Wait, s=4 means v and w agree on all 4 positions, so v=w. That's the same vector.
    #   Actually for m=4, s can be 0 to 4. 2s>6 means s>=4, so s=4 only.
    #   But s=4 means v=w exactly. So distinct vectors with m=4 don't conflict?
    #   Wait, if m=4 and s=3, then 2*3-4 = 2, so v·w=2, dist²=8-4=4, not < 4. OK.
    #   If m=4 and s=4, v·w=4, dist²=8-8=0 < 4. But that's the same vector.
    # So for DISTINCT vectors: conflict iff m=3 and s=3 (all 3 shared positions have same sign)
    # 
    # Let me verify: v and w share exactly 3 positions, and on those 3 positions, v_i = w_i.
    # Then v·w = 3 (from the 3 matching positions) + 0 (from the 1th position of v not in w) 
    #   + 0 (from the 1th position of w not in v) = 3.
    # dist² = 8 - 2*3 = 2 < 4. Conflict! ✓
    #
    # If m=3 and s=2: v·w = 2-1 = 1. dist² = 8-2 = 6 >= 4. No conflict.
    # If m=3 and s=1: v·w = 1-2 = -1. dist² = 8+2 = 10 >= 4. No conflict.
    # If m=3 and s=0: v·w = 0-3 = -3. dist² = 8+6 = 14 >= 4. No conflict.
    #
    # So the conflict condition for distinct ±1 weight-4 vectors is:
    # They share exactly 3 support positions, AND on all 3 shared positions they have the same sign.
    
    # Generate all 5280 vectors
    all_vecs = []
    vec_to_idx = {}
    for support in combinations(range(d), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for pos, sign in zip(support, signs):
                v[pos] = sign
            key = tuple(v)
            idx = len(all_vecs)
            all_vecs.append(v)
            vec_to_idx[key] = idx
    
    n_total = len(all_vecs)  # 5280
    
    # Build conflict graph
    # For each vector, find all others that conflict with it
    # Two vectors conflict iff they share exactly 3 support positions with all same signs
    # 
    # Efficient approach: For each vector v with support S and signs,
    # any conflicting vector w must have support T where |T∩S|=3 and on T∩S, w agrees with v.
    # T = (T∩S) ∪ {t} where t ∉ S, and |T∩S|=3.
    # So T = (S \ {s}) ∪ {t} for some s∈S, t∉S, and w agrees with v on S\{s}.
    # 
    # For a given v with support S and signs:
    # For each choice of which 3 positions to keep (i.e., which 1 to drop):
    #   Let S3 = S \ {s} for s in S (4 choices)
    #   Let signs3 = signs on S3
    #   For each t in [11] \ S (7 choices):
    #     T = S3 ∪ {t}
    #     w must have signs = signs3 on S3, and ±1 on t (2 choices)
    #     So 2 vectors per (s, t) pair
    #   Total per s: 7*2 = 14 vectors
    #   Total: 4*14 = 56 potential conflicts per vector
    
    conflict_adj = defaultdict(set)
    for idx, v in enumerate(all_vecs):
        # Find support and signs
        support = tuple(i for i in range(d) if v[i] != 0)
        signs = tuple(v[i] for i in support)
        
        for drop_idx in range(4):
            s = support[drop_idx]
            S3 = support[:drop_idx] + support[drop_idx+1:]
            signs3 = signs[:drop_idx] + signs[drop_idx+1:]
            
            for t in range(d):
                if t == s:
                    continue
                # T = S3 ∪ {t}
                # w has signs3 on S3 and ±1 on t
                for st in [1, -1]:
                    w = np.zeros(d, dtype=np.int64)
                    for pos, sign in zip(S3, signs3):
                        w[pos] = sign
                    w[t] = st
                    key = tuple(w)
                    if key in vec_to_idx:
                        jdx = vec_to_idx[key]
                        if jdx != idx:
                            conflict_adj[idx].add(jdx)
                            conflict_adj[jdx].add(idx)
    
    # Now solve MWIS on this graph (all weights = 1, so it's Maximum Independent Set)
    # Use branch and bound with reduction rules
    
    # Convert to adjacency list for faster access
    adj = [list(conflict_adj[i]) for i in range(n_total)]
    
    # Greedy initial solution: pick vertices in order of lowest degree first
    # Actually, let's use a greedy coloring bound approach
    
    # Branch and bound for Maximum Independent Set
    best_set = []
    best_size = 0
    
    # Use a recursive branch and bound
    # State: current independent set, candidate vertices (those not in set and not adjacent to any in set)
    
    # For efficiency, use bitsets
    # n_total = 5280, which is too large for a single Python int bitmask in practice
    # Let's use a different approach: greedy + local search, or a more structured B&B
    
    # Actually, let's try a different structural approach first.
    # The conflict graph has a specific structure. Let's analyze it.
    # 
    # Each vertex is a (support, signs) pair. Two vertices conflict iff they share 
    # exactly 3 support positions with all same signs.
    # 
    # Alternative: think of it as a hypergraph or use a different decomposition.
    # 
    # Let me try: group by support. There are C(11,4)=330 supports, each with 16 sign patterns.
    # Two vertices with the same support never conflict (they share 4 positions, need s=4 to conflict,
    # which means same vector).
    # Two vertices with supports sharing 4 positions: same support, no conflict (for distinct).
    # Two vertices with supports sharing 3 positions: conflict iff same signs on those 3 positions.
    # Two vertices with supports sharing 2 or fewer: no conflict.
    
    # So the conflict structure is:
    # For each triple of positions (there are C(11,3)=165 triples) and each sign pattern on those 3 positions (8 patterns):
    # All vectors that contain that triple with that sign pattern are mutually conflicting IF they share exactly those 3.
    # Wait, no. Two vectors conflict iff they share exactly 3 positions with same signs.
    # If two vectors share 3 positions with same signs AND also share a 4th position, they share 4 positions.
    # But we established that sharing 4 positions doesn't cause conflict for distinct vectors.
    # So the conflict is specifically: share exactly 3 positions, same signs on those 3.
    
    # Let me try a greedy approach with careful ordering, then local search.
    
    # Greedy: order by degree (ascending), pick vertex if no conflict with already picked
    degrees = [len(adj[i]) for i in range(n_total)]
    order = sorted(range(n_total), key=lambda i: degrees[i])
    
    greedy_set = set()
    for idx in order:
        if all(j not in greedy_set for j in adj[idx]):
            greedy_set.add(idx)
    
    best_size = len(greedy_set)
    best_set = list(greedy_set)
    
    # Local search: try to improve by swapping
    if elapsed() < deadline - 100:
        current_set = set(greedy_set)
        current_size = len(current_set)
        
        # Try adding vertices not in set
        # For each vertex not in set, check if it can be added
        # If not, try removing one of its neighbors and adding it
        
        improved = True
        while improved and elapsed() < deadline - 100:
            improved = False
            # Try to add each non-set vertex
            for idx in range(n_total):
                if idx in current_set:
                    continue
                # Check if idx conflicts with any in current_set
                conflicting = [j for j in adj[idx] if j in current_set]
                if len(conflicting) == 0:
                    current_set.add(idx)
                    current_size += 1
                    improved = True
                elif len(conflicting) == 1:
                    # Try removing the one conflicting vertex
                    # Only beneficial if we can add more than we remove, but here it's 1-for-1
                    # So no gain. Skip.
                    pass
                elif len(conflicting) == 2:
                    # Try removing both and adding idx: net -1. Not helpful unless we can add more.
                    # For now, skip.
                    pass
                # Continue to try all vertices for direct additions
        
        if current_size > best_size:
            best_size = current_size
            best_set = list(current_set)
    
    # Try a more sophisticated approach: random restarts with greedy
    if elapsed() < deadline - 200:
        import random
        rng = random.Random(42)
        
        for restart in range(50):
            if elapsed() > deadline - 200:
                break
            
            # Randomize order slightly
            order = list(range(n_total))
            rng.shuffle(order)
            # Sort by degree within random order
            order.sort(key=lambda i: (degrees[i], rng.random()))
            
            greedy_set = set()
            for idx in order:
                if all(j not in greedy_set for j in adj[idx]):
                    greedy_set.add(idx)
            
            if len(greedy_set) > best_size:
                best_size = len(greedy_set)
                best_set = list(greedy_set)
    
    # Build the result
    if best_size >= 2:
        result_pts = np.array([all_vecs[i] for i in best_set], dtype=np.int64)
    else:
        # Fallback: just use unit vectors
        result_pts = np.array([np.eye(d, dtype=np.int64)[i] * s for i in range(d) for s in [1, -1]])
    
    # Verify
    n_pts = len(result_pts)
    norms = np.sum(result_pts ** 2, axis=1)
    max_norm = int(norms.max())
    
    min_dist = float('inf')
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            diff = result_pts[i] - result_pts[j]
            dist_sq = int(np.sum(diff ** 2))
            if dist_sq < min_dist:
                min_dist = dist_sq
    
    if min_dist < max_norm or n_pts < 2:
        # Fallback
        result_pts = np.array([np.eye(d, dtype=np.int64)[i] * s for i in range(d) for s in [1, -1]])
    
    return np.around(result_pts).astype(np.int64)
