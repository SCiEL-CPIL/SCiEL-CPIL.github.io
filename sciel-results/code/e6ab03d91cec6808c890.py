"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct():
    """
    Construct integer vectors in R^11 for the kissing number problem.
    
    Strategy: Generate all nonzero integer vectors of norm 2 (squared norm 4).
    These come in two types:
    Type A: One coordinate is ±2, others 0. (11 * 2 = 22 vectors)
    Type B: Four coordinates are ±1, others 0. (C(11,4) * 2^4 = 330 * 16 = 5280 vectors)
    Total: 5302 vectors.
    
    Verification:
    - Max norm = 2 for all vectors.
    - Min pairwise distance: 
      * Between Type A vectors: distance >= 2 (if same coordinate, dist=0 but they're ±2 so dist=4 or 0? No, ±2 vs ∓2 is dist 4, same is 0 but we don't have duplicates. Actually e_i*2 and -e_i*2 have distance 4. Different coordinates: dist^2 = 4+4=8, dist=sqrt(8)>=2).
      * Between Type B vectors: If they share k coordinates, distance squared = 2*(4-k) + ... Actually, let's think.
        Two Type B vectors with supports S and T. |S|=|T|=4.
        If S=T, then each coordinate has signs ±1. Distance squared = sum (s_i - t_i)^2. Each term is 0 or 4. Min nonzero is 4 (one sign flipped). So dist=2.
        If S != T, they share k coordinates (k=0,1,2,3). On shared coordinates, diff is 0 or 4. On non-shared, one is 0, other is ±1, diff is 1.
        We need min distance >= 2 (dist^2 >= 4).
        If k=3: share 3, differ in 1 on each side. 
        Shared coords: if signs match, diff 0. If mismatch, diff 4.
        Non-shared: 1 coord in S\T, 1 in T\S. Diff^2 = 1 + 1 = 2.
        Total dist^2 = (sum of diffs on shared) + 2.
        To minimize, we want shared diffs to be 0. Then dist^2 = 2 < 4. 
        Wait, this implies distance sqrt(2) < 2.
        Let's check: v1 = (1,1,1,1,0...), v2 = (1,1,1,-1,0...)? No, if k=3, they share 3 indices.
        Example: v1 supported on {0,1,2,3} with signs (1,1,1,1). v2 supported on {0,1,2,4} with signs (1,1,1,1).
        v1 = [1,1,1,1,0,0,0,0,0,0,0]
        v2 = [1,1,1,0,1,0,0,0,0,0,0]
        Diff = [0,0,0,1,-1,0,0,0,0,0,0]
        Dist^2 = 1 + 1 = 2. Dist = sqrt(2) ≈ 1.414 < 2.
        
        So Type B vectors do NOT all pairwise satisfy dist >= 2.
        The set of ALL norm 2 integer vectors is NOT a valid kissing configuration.
        
        Re-evaluating the task:
        "minimum pairwise distance is at least the maximum vector norm"
        If we include vectors where dist < max_norm, it's invalid.
        
        The D11 lattice kissing number is 2304? Or is it related to E8 or something?
        In dimension 11, the known kissing number is 2304? 
        Actually, the kissing number for d=11 is 2304? No, that's d=8 (240) or d=12 (2560).
        For d=11, it's known to be at least 2304? 
        Wait, the Leech lattice is d=24.
        
        Let's look at the D_n lattice.
        Shortest vectors in D_n are (±1, ±1, 0...0) with even sum of coordinates?
        No, D_n is {(x_i) in Z^n : sum x_i is even}.
        Shortest vectors have squared norm 2.
        They are of form (±1, ±1, 0, ..., 0).
        Number of such vectors: C(n,2) * 4.
        For n=11: C(11,2)*4 = 55*4 = 220.
        This matches the seed program!
        
        The seed program generates these 220 vectors.
        Max norm = sqrt(2).
        Min distance:
        Two vectors (1,1,0..) and (1,-1,0..): dist^2 = 0 + 4 + 0... = 4. dist=2.
        2 >= sqrt(2). OK.
        Two vectors (1,1,0..) and (1,0,1..): dist^2 = 0 + 1 + 1 = 2. dist=sqrt(2).
        sqrt(2) >= sqrt(2). OK.
        Two vectors (1,1,0..) and (-1,1,0..): dist^2 = 4 + 0 ... = 4. dist=2. OK.
        
        So the 220 vectors of D_11 form a valid configuration.
        Score = 220.
        
        Can we do better?
        The task asks for "integer vectors".
        If we scale the D_11 vectors by 2?
        v = 2*(±1, ±1, 0..0).
        Norm = 2*sqrt(2) = sqrt(8).
        Min distance:
        If original dist was d, new dist is 2d.
        Min original dist = sqrt(2). New min dist = 2*sqrt(2) = sqrt(8).
        Max norm = sqrt(8).
        Condition: min_dist >= max_norm.
        sqrt(8) >= sqrt(8). OK.
        Number of vectors is still 220.
        
        What if we add more vectors?
        The kissing number of R^11 is not known exactly, but lower bounds are high.
        However, we are restricted to INTEGER vectors.
        
        Let's check the lattice A_11 or something?
        
        Actually, the set of all (±1, ±1, 0...0) is the standard construction.
        Is there a larger set of integer vectors with min_dist >= max_norm?
        
        Consider the set of all vectors with 4 non-zero coordinates of ±1.
        We saw that two such vectors can be at distance sqrt(2).
        If we take a subset of these that are pairwise distance >= sqrt(2)?
        But max norm is sqrt(4)=2.
        We need dist >= 2.
        So we need dist^2 >= 4.
        For two vectors with 4 non-zeros:
        If they share 3 coordinates: dist^2 can be 2 (as shown).
        If they share 2 coordinates: dist^2 = (sum diffs on 2 shared) + (1+1 for the two unique on each side? No, 2 unique on S, 2 on T).
        Let's calculate dist^2 for two Type B vectors (4 non-zeros).
        Intersection size k.
        Non-intersection: 4-k on each.
        Dist^2 = (sum_{i in S∩T} (s_i - t_i)^2) + (sum_{i in S\T} 1^2) + (sum_{i in T\S} 1^2)
        = (sum_{i in S∩T} (s_i - t_i)^2) + (4-k) + (4-k)
        = (sum_{i in S∩T} (s_i - t_i)^2) + 8 - 2k.
        
        We need dist^2 >= 4.
        So (sum_{i in S∩T} (s_i - t_i)^2) >= 2k - 4.
        
        If k=4 (same support): sum >= 4.
        Each term is 0 or 4. So at least one sign must differ.
        If k=3: sum >= 2.
        One term is 4 (if signs differ) or 0. So at least one sign must differ.
        If k=2: sum >= 0. Always true.
        If k=1: sum >= -2. Always true.
        If k=0: sum >= -4. Always true.
        
        So the condition fails only if:
        k=4 and all signs same (identical vectors - not an issue as we consider distinct pairs)
        k=3 and all signs same on intersection (then sum=0, 0 < 2, fail)
        
        So we cannot have two vectors with same support and same signs (duplicate).
        We cannot have two vectors sharing 3 coordinates with same signs on those 3.
        
        This suggests we can construct a large set from Type B vectors by ensuring no two share 3 coordinates with same signs.
        
        This is a coding theory problem.
        We want a code of length 11, size 4 (support), alphabet {0,1,-1}, distance >= 2 (in terms of L2 squared >= 4? No, the condition is derived).
        
        Actually, let's look at the "even" constraint.
        The D_n lattice requires sum of coordinates to be even.
        Type B vectors (four ±1s) have sum = even (±4, ±2, 0).
        So they are in D_11.
        But D_11 shortest vectors are only the 2-nonzero ones.
        The 4-nonzero ones have norm 2, which is longer.
        
        If we take ALL vectors in D_11 of norm 2 (squared 4)?
        That includes:
        1. Four ±1s (5280 vectors).
        2. One ±2 (22 vectors).
        
        Do they satisfy the condition?
        We found that two 4-nonzero vectors can be too close.
        What about a 4-nonzero and a 1-nonzero (±2)?
        v1 = (1,1,1,1,0..). v2 = (2,0,0,0,0..).
        v1 support {0,1,2,3}, v2 support {0}.
        Intersection {0}.
        v1[0]=1, v2[0]=2. Diff = -1. Sq=1.
        Other v1 coords: 1,1,1. Diff = -1, -1, -1. Sq=1,1,1.
        Total dist^2 = 1 (at 0) + 1+1+1 (at 1,2,3) = 4.
        Dist = 2.
        Max norm = 2.
        2 >= 2. OK.
        
        So the 22 Type A vectors are compatible with Type B vectors?
        Let's check v1=(1,1,1,1,0..), v2=(-2,0,0,0,0..).
        v1[0]=1, v2[0]=-2. Diff=3. Sq=9.
        Others 1. Sq=1,1,1.
        Total 12. Dist=sqrt(12) > 2. OK.
        
        So the incompatibility is only between Type B vectors sharing 3 coordinates with same signs.
        
        If we take the set of all 5302 vectors, is it valid?
        No, because of the Type B - Type B collision.
        
        How many collisions?
        We need to remove vectors to fix this.
        Or, maybe the 220 D_n vectors are the intended "kissing number" in this integer lattice context?
        The kissing number of D_11 is 220.
        
        The task says "Kissing number in dimension 11".
        Usually this refers to the maximum number of unit spheres that can touch a central unit sphere.
        In R^11, this number is not known exactly, but 2304 is a lower bound?
        Wait, 2304 is the kissing number for d=8? No, d=8 is 240.
        d=12 is 2560.
        d=11 is at least 2304?
        Actually, the kissing number for d=11 is 2304?
        Let me check.
        Kissing numbers:
        d=1: 2
        d=2: 6
        d=3: 12
        d=4: 24
        d=5: 40
        d=6: 72
        d=7: 126
        d=8: 240
        d=9: 400 (known)
        d=10: 576?
        d=11: 784?
        
        Wait, the D_n lattice kissing number is 2*n*(n-1).
        For n=11, 2*11*10 = 220.
        
        If the task implies the D_n lattice structure, then 220 is the answer.
        But the task says "Construct nonzero integer vectors...".
        It doesn't restrict to the D_n lattice.
        
        However, if we can construct 5302 vectors (or a subset thereof) that satisfies the condition, the score would be much higher.
        
        Let's try to construct a valid subset of the 5302 vectors.
        We need to resolve the conflicts between Type B vectors.
        Conflict: Two Type B vectors with same support and same signs on 3 coordinates.
        Actually, if they have same support and same signs, they are identical.
        If they have same support and 3 signs same, 1 sign different: dist^2 = 4. OK.
        If they have different support (share 3 coords), and signs on shared coords are same: dist^2 = 2. FAIL.
        
        So we cannot have v1 and v2 where:
        - Supports S1, S2 with |S1|=|S2|=4, |S1∩S2|=3.
        - For all i in S1∩S2, v1[i] == v2[i].
        
        This is a constraint satisfaction problem.
        We can model this as a graph where nodes are the 5280 Type B vectors, and edges connect incompatible pairs.
        We want a Maximum Independent Set.
        
        The graph is large (5280 nodes).
        But it has structure.
        
        Alternative:
        Just use the 220 D_11 vectors.
        Score = 220.
        
        Can we improve?
        What if we use all 22 Type A vectors?
        They are mutually compatible (dist >= 2, norm sqrt(8)? No, norm 2).
        Wait, Type A vectors are (±2, 0..0). Norm 2.
        Dist between (2,0..) and (2,0..) is 0 (same vector).
        Dist between (2,0..) and (-2,0..) is 4.
        Dist between (2,0..) and (0,2..) is sqrt(8).
        So the 22 Type A vectors form a valid set.
        
        Can we combine Type A and a subset of Type B?
        We established Type A and Type B are compatible.
        
        So we can add the 22 Type A vectors to any valid subset of Type B.
        
        The question is: how large a valid subset of Type B can we get?
        
        If we just take one sign assignment per support?
        There are 330 supports.
        For each support, there are 16 sign assignments.
        We can pick 1 assignment per support.
        Size = 330.
        Plus 22 Type A.
        Total 352.
        
        Are two vectors from different supports compatible?
        If they share 3 coordinates, we need signs to differ on at least one shared coordinate.
        If we pick arbitrary signs, they might conflict.
        
        Example:
        Support {0,1,2,3}: signs (1,1,1,1).
        Support {0,1,2,4}: signs (1,1,1,1).
        Shared {0,1,2}. Signs (1,1,1) vs (1,1,1). Same.
        Conflict!
        
        So we need to choose signs such that for any two supports sharing 3 elements, the signs on the intersection are not identical.
        
        This is like a coloring problem or code construction.
        
        If we choose signs randomly, the probability of conflict is low?
        For a pair sharing 3 coords, there are 2^3 = 8 sign patterns on intersection.
        We need to avoid the case where patterns are identical.
        
        Actually, if we pick signs uniformly at random for each support, the probability that two specific supports (sharing 3) have identical signs on the intersection is (1/2)^3 = 1/8.
        
        We can use a greedy algorithm to pick a large independent set.
        
        Given the time limit, a greedy approach on the 5280 nodes might be feasible.
        5280 nodes.
        Edges: Each node is incompatible with nodes sharing 3 coords with same signs.
        
        Let's implement a greedy MIS on the Type B vectors, then add Type A.
        
        Steps:
        1. Generate all 5280 Type B vectors.
        2. Generate all 22 Type A vectors.
        3. Build adjacency or check compatibility on the fly.
        4. Greedy selection:
           - Start with empty set.
           - Iterate through Type B vectors.
           - If compatible with all currently selected Type B vectors, add it.
        5. Add all Type A vectors (they are compatible with each other and with Type B).
        6. Return the combined set.
        
        Complexity:
        5280 vectors.
        For each, check against selected set.
        Worst case O(N^2) = 27M checks.
        Each check is O(11).
        300M ops. Might be slow in Python but maybe OK with NumPy vectorization?
        
        Better:
        Group by support?
        No, conflicts are between different supports.
        
        Let's try a simpler heuristic first:
        Just take the 220 D_11 vectors.
        Score 220.
        
        If we can get 352+ with the greedy, that's a 60% improvement.
        
        Let's try to implement the greedy.
        
        Optimization:
        Precompute the "signature" of the 3-coordinate intersections?
        There are C(11,3) = 165 triples.
        For each Type B vector, it covers 4 triples (one for each coordinate removed).
        For each triple, it has a specific sign pattern (3 bits).
        
        Conflict condition:
        Two vectors v1, v2 conflict if they share a triple T such that their sign patterns on T are identical.
        
        So we can maintain a set of "occupied" (triple, sign_pattern) pairs.
        If a new vector has any (triple, sign_pattern) already occupied, it conflicts.
        
        This is much faster!
        
        Algorithm:
        1. Generate Type B vectors.
        2. For each vector, compute its 4 (triple, sign_pattern) keys.
        3. Sort vectors by... maybe just random order?
        4. Iterate:
           - If none of its keys are in the occupied set, add to result, mark keys as occupied.
           - Else, skip.
        
        This finds a maximal independent set (not necessarily maximum, but likely large).
        
        Expected size:
        Total keys = 165 * 8 = 1320 possible (triple, pattern) states.
        Each vector uses 4 states.
        We can pack at most 1320/4 = 330 vectors if no overlaps in keys?
        But vectors can share keys if they don't conflict?
        No, if they share a key, they conflict.
        So the keys must be disjoint.
        
        So the maximum size is bounded by 330.
        Plus 22 Type A.
        Total 352.
        
        So the greedy should find close to 330.
        
        This is an improvement over 220.
        
        Let's implement this.

    """
    deadline = time.time() + 140  # Safety margin

    # Type A: (±2, 0, ..., 0)
    type_a = []
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            type_a.append(v)
    
    # Type B: four ±1s
    type_b_vectors = []
    # Precompute triples
    triples = list(itertools.combinations(range(D), 3))
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for k, idx in enumerate(combo):
                v[idx] = signs[k]
            type_b_vectors.append(v)
            
    # Greedy selection for Type B
    occupied = set()
    selected_b = []
    
    # Shuffle to avoid bias? Or just iterate.
    # Let's iterate in a deterministic order but maybe randomize for better spread?
    # For now, deterministic.
    for v in type_b_vectors:
        if time.time() > deadline:
            break
            
        keys = []
        # Find the 4 triples (remove one coordinate from the 4-support)
        # Support is the indices where v is non-zero
        support = [i for i in range(D) if v[i] != 0]
        for i in range(4):
            # Triple is support without i-th element
            t = tuple(support[j] for j in range(4) if j != i)
            # Sign pattern on this triple
            # Need to map signs to a canonical representation
            # Let's use a tuple of signs
            sig = tuple(v[idx] for idx in t)
            key = (triple_to_idx[t], sig)
            keys.append(key)
        
        # Check conflict
        conflict = False
        for key in keys:
            if key in occupied:
                conflict = True
                break
        
        if not conflict:
            for key in keys:
                occupied.add(key)
            selected_b.append(v)
            
    # Combine
    all_vectors = type_a + selected_b
    if not all_vectors:
        return np.array([[1] + [0]*(D-1)], dtype=np.int64)
        
    return np.array(all_vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
