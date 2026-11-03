import numpy as np
import random
import itertools
import time

def construct():
    D = 11
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)
    
    # Total number of vectors: 16 per support
    # Node ID = sup_idx * 16 + sign_idx
    # sign_idx corresponds to the 4 bits of the signs
    total_nodes = n_sup * 16
    
    # Precompute signs for each node
    # signs[sign_idx] is a tuple of 4 values +/-1
    sign_tuples = list(itertools.product([1, -1], repeat=4))
    
    # Map support index to set of indices for fast intersection?
    # Or just use the tuple.
    
    # Build conflict graph
    # adj[i] is a set of j such that dist^2(u_i, u_j) < 4
    # We only need to check pairs where supports intersect in 3 or 4.
    # If intersect 4 (same support):
    #   d^2 = sum (u_i - v_i)^2.
    #   If signs are same, d=0 (same node, ignore).
    #   If 1 sign diff: d^2 = 4. (Valid, d>=2)
    #   If 2 sign diff: d^2 = 8.
    #   So for same support, only identical sign is conflict (self-loop, ignore).
    #   Wait, if signs are same, it's the same vector. We don't add self-loops.
    #   So nodes with same support are NEVER conflicting with each other?
    #   Let's check: u = (1,1,1,1), v = (1,1,1,-1). d^2 = 0+0+0+4 = 4. d=2. Valid.
    #   So we can pick ALL 16 sign variations for a single support?
    #   Yes! They are all mutually distance 2 or more apart.
    #   This immediately gives us 16 vectors for any single support.
    #   The current program picks at most 1 vector per support (actually it picks 1 vector per support index in the independent set, but then expands to all 16 signs in the output? 
    #   LOOK AT CURRENT CODE:
    #   `for idx in best_set:` ... `for signs in itertools.product([1, -1], repeat=4):` ... `vectors.append(v)`
    #   The current code DOES output all 16 signs for each selected support.
    #   BUT, it treats the supports as the graph nodes.
    #   It assumes if supports intersect in 3, they conflict.
    #   As analyzed, they only conflict if signs are identical on the intersection.
    #   So the current code is extremely conservative. It finds an independent set of *supports*, then multiplies by 16.
    #   If we can find a larger independent set of supports, or better, allow overlapping supports with different signs, we can do much better.
    
    # However, if we just output all 16 signs for a set of supports, we MUST ensure that for ANY two supports S1, S2 in the set, and ANY sign combination s1, s2, the distance is valid.
    # This is NOT true.
    # If S1 and S2 intersect in 3, there EXISTS sign combinations (identical on intersection) that are invalid.
    # The current code's output is INVALID if the supports intersect in 3!
    # Let's re-read the task: "minimum pairwise distance is at least the maximum vector norm".
    # If the current code outputs all 16 signs for two supports that intersect in 3, it includes the pair with identical signs on the intersection, which has distance 2 (norm 2).
    # Wait. Distance 2 is "at least" norm 2.
    # So d=2 is VALID.
    # My previous analysis: "If they agree on all 3, d^2 = 2 < 4. Conflict."
    # Let's re-calculate.
    # u, v weight 4.
    # Intersection 3.
    # u = (1, 1, 1, 1, 0, ...)
    # v = (1, 1, 1, 0, 1, ...)
    # u-v = (0, 0, 0, 1, -1, ...)
    # d^2 = 1 + 1 = 2.
    # Norm of u is 2. Norm squared is 4.
    # Condition: d >= norm(u). d^2 >= 4.
    # Here d^2 = 2. 2 < 4. INVALID.
    # So the current code IS producing invalid configurations if it selects two supports with intersection 3, because it outputs the "bad" sign pairs.
    # BUT the current code's SA operates on the support graph. It ensures no two supports in `best_set` intersect in 3.
    # If no two supports intersect in 3, then for any pair of supports, intersection is <= 2.
    # If intersection <= 2, min d^2 is 4.
    # So the current code's output is VALID.
    # But it is very restrictive. It forbids any pair of supports with intersection 3, even though 7/8 of the sign pairs are valid.
    
    # To improve, we should model the graph on the 5280 nodes (support, sign).
    # Find a large independent set in this node graph.
    # Each node is a specific vector.
    # This allows us to pick multiple supports that intersect in 3, as long as we avoid the specific sign combinations that conflict.
    
    # Let's build the graph.
    # Node i: (sup_idx, sign_idx)
    # Conflict if d^2 < 4.
    
    # Optimization:
    # Only check pairs where sup_intersect >= 3.
    # If sup_intersect == 4 (same support):
    #   d^2 is 0 (same sign) or 4, 8, 12.
    #   Only same sign is d=0. But we don't have self-loops.
    #   So nodes with same support are NEVER conflicting.
    # If sup_intersect == 3:
    #   Let intersection indices be I (size 3).
    #   u has value a_i at I, v has value b_i at I.
    #   d^2 = (1-0)^2 + (0-1)^2 + sum_{i in I} (a_i - b_i)^2
    #   d^2 = 2 + sum (a_i - b_i)^2.
    #   a_i, b_i are +/- 1.
    #   (a-b)^2 is 0 if a==b, 4 if a!=b.
    #   Let k be number of positions in I where a_i != b_i.
    #   d^2 = 2 + 4k.
    #   We need d^2 >= 4 => 2 + 4k >= 4 => 4k >= 2 => k >= 0.5 => k >= 1.
    #   So conflict if k=0, i.e., a_i == b_i for all i in I.
    #   So u, v conflict iff their signs on the 3 intersection coordinates are identical.
    
    # Graph Construction:
    # For each pair of supports (S1, S2) with intersection size 3:
    #   Let I be the intersection.
    #   For each sign combination s1 (4 bits) and s2 (4 bits):
    #     Check if s1 restricted to I == s2 restricted to I.
    #     If yes, add edge between node (S1, s1) and node (S2, s2).
    
    # This graph has 5280 nodes.
    # Edges:
    #   For each pair of supports with int 3:
    #   There are 16x16=256 pairs of signs.
    #   How many conflict?
    #   We need s1[I] == s2[I].
    #   There are 2^3 = 8 possible patterns for s1[I].
    #   For a fixed pattern, s1 can be any of 2^(4-3)=2 variations on the 4th bit? No, s1 is defined on 4 positions. 3 are in I, 1 is outside.
    #   So for a fixed pattern on I, there are 2 choices for the outside bit.
    #   So 2 * 2 = 4 pairs of signs (s1, s2) conflict for a fixed pattern?
    #   Total conflicts per support pair = 8 patterns * 2 (s1 ext) * 2 (s2 ext) = 32 edges.
    #   Number of support pairs with int 3:
    #   Choose 3 from 11: C(11,3) = 165.
    #   For each set of 3, how many supports contain them?
    #   The 4th element can be any of the remaining 8.
    #   So C(8,1) = 8 supports for each intersection set?
    #   No, a support is a set of 4.
    #   Pairs of supports with intersection exactly 3:
    #   Total pairs of supports: C(330, 2) ~ 54k.
    #   This is small enough to build explicitly.
    
    # Let's build the adjacency list.
    adj = [set() for _ in range(total_nodes)]
    
    # Map support to index
    sup_to_idx = {sup: i for i, sup in enumerate(supports)}
    
    # Precompute intersection of 3 for each pair of supports?
    # Iterate over all pairs of supports.
    for i in range(n_sup):
        s_i = set(supports[i])
        for j in range(i+1, n_sup):
            s_j = set(supports[j])
            inter = s_i.intersection(s_j)
            if len(inter) == 3:
                # Determine which position in the 4-tuple corresponds to the intersection
                # supports[i] is a tuple of 4 sorted indices.
                # We need to map the intersection indices to positions 0,1,2,3 in the sign tuple.
                
                # Create a map: index -> position in support i
                pos_map_i = {idx: k for k, idx in enumerate(supports[i])}
                pos_map_j = {idx: k for k, idx in enumerate(supports[j])}
                
                # Get positions of intersection in i and j
                # inter is a set of 3 indices.
                # We need to know which sign bits correspond to these indices.
                
                # For each sign combination s1 (0-15) and s2 (0-15):
                # Node i1 = i * 16 + s1
                # Node j1 = j * 16 + s2
                
                # Check if signs on intersection are same.
                # Extract signs for intersection from s1, s2.
                
                # To speed this up, precompute for each support pair the "conflict mask" or just iterate.
                # 32 conflicts per pair.
                # Number of pairs with int 3:
                # Let's estimate.
                # C(11,3) = 165.
                # For a fixed set of 3 indices, there are 8 supports containing them.
                # Pairs of supports from these 8: C(8,2) = 28.
                # Total pairs = 165 * 28 = 4620.
                # Each pair has 32 edges.
                # Total edges ~ 4620 * 32 = 147,840.
                # This is very small. We can build it fast.
                
                for s1 in range(16):
                    # Get sign bits for s1 at positions corresponding to inter
                    # We need to check if s1[inter_pos_i] == s2[inter_pos_j] for all 3.
                    pass # We'll do this in the loop below
    
    # Actually, let's just do the nested loop with a quick check.
    # To make it faster, we can precompute the "intersection pattern" for each support pair.
    
    # Let's restructure for speed.
    # For each pair (i, j) with intersection 3:
    #   Let P_i be the 3 positions in support i that are in the intersection.
    #   Let P_j be the 3 positions in support j that are in the intersection.
    #   For s1 in 0..15:
    #      bits1 = (s1 >> P_i[0]) & 1, ...
    #      This is messy.
    
    # Alternative:
    # For a fixed support i, and a fixed intersection set I (size 3) with another support j:
    # The condition is that the 3 bits of s1 at positions corresponding to I must equal the 3 bits of s2 at positions corresponding to I.
    
    # Let's just iterate. 148k edges is nothing.
    
    for i in range(n_sup):
        s_i = set(supports[i])
        pos_i = {idx: k for k, idx in enumerate(supports[i])}
        for j in range(i+1, n_sup):
            s_j = set(supports[j])
            inter = s_i.intersection(s_j)
            if len(inter) == 3:
                pos_j = {idx: k for k, idx in enumerate(supports[j])}
                
                # Map intersection indices to positions in i and j
                # inter is a set. We need to iterate over it.
                # But we need to check equality of bits.
                # Let's extract the 3 bits for each sign.
                
                # Precompute bit masks?
                # For support i, the positions of the 3 intersection elements.
                # Let's say they are at positions p1, p2, p3 in support i.
                # And q1, q2, q3 in support j.
                
                # We can just loop s1, s2.
                # 16*16 = 256 iterations per pair.
                # 4620 pairs * 256 = 1.18 million iterations.
                # In Python, this might take a second or two. Acceptable.
                
                # Get positions
                # inter is a set of 3 indices from 0..10.
                # pos_i[idx] gives the index in the 4-tuple.
                
                # We need to compare the bits at these positions.
                # To avoid creating lists of positions repeatedly, we can just do it.
                
                # Optimization: Group signs by their projection on the intersection.
                # For support i, project each sign s1 (0-15) onto the 3 intersection bits.
                # There are 8 possible projections.
                # Map projection -> list of s1.
                
                proj_i = {}
                for s1 in range(16):
                    # Extract bits at positions in inter
                    # We need the positions of inter in support i.
                    # pos_i is a dict.
                    # Let's get the 3 positions.
                    # Since inter is small, just iterate.
                    # But we need to be careful with bit shifting.
                    # s1 is an integer 0-15. Bit k corresponds to position k.
                    # We want bits at positions pos_i[inter_elem].
                    
                    # Let's precompute the positions for this pair.
                    # This is done inside the j loop, so we can do it once per pair.
                    pass
                
                # Let's do it simply first.
                # Get the 3 positions in i
                # inter_list = list(inter)
                # p1, p2, p3 = [pos_i[x] for x in inter_list]
                # q1, q2, q3 = [pos_j[x] for x in inter_list]
                
                # Then for s1, s2:
                # if ((s1>>p1)&1 == (s2>>q1)&1) and ...:
                #    add edge
                
                # Let's optimize by precomputing the positions for the pair.
                inter_list = list(inter)
                p_pos = [pos_i[x] for x in inter_list]
                q_pos = [pos_j[x] for x in inter_list]
                
                # Precompute projection for s1
                # proj1[s1] = tuple of 3 bits
                # But we can just compute on the fly.
                
                for s1 in range(16):
                    node_i = i * 16 + s1
                    # Extract bits
                    b1_1 = (s1 >> p_pos[0]) & 1
                    b1_2 = (s1 >> p_pos[1]) & 1
                    b1_3 = (s1 >> p_pos[2]) & 1
                    
                    for s2 in range(16):
                        node_j = j * 16 + s2
                        b2_1 = (s2 >> q_pos[0]) & 1
                        b2_2 = (s2 >> q_pos[1]) & 1
                        b2_3 = (s2 >> q_pos[2]) & 1
                        
                        if b1_1 == b2_1 and b1_2 == b2_2 and b1_3 == b2_3:
                            adj[node_i].add(node_j)
                            adj[node_j].add(node_i)

    # Now we have the exact conflict graph.
    # Run SA to find large independent set.
    
    # Initial solution: Random or Greedy.
    # Greedy: Pick random node, add to set, remove neighbors. Repeat.
    
    def greedy_is():
        remaining = set(range(total_nodes))
        current_set = set()
        # Randomize order to get different starts?
        # Or just pick min degree.
        while remaining:
            # Pick node with min degree in remaining
            min_deg = float('inf')
            best_v = None
            # Sample a few to speed up? No, 5280 nodes is small.
            # But 5280 * 5280 is 27M ops. Might be slow in Python.
            # Let's just pick a random node from remaining?
            # Or maintain a list?
            # Let's use random choice for speed in initialization.
            # Actually, min-degree is better for IS size.
            # Let's try to speed up min-degree.
            
            # Alternative: Just pick a random node.
            v = random.choice(list(remaining))
            current_set.add(v)
            to_remove = {v}.union(adj[v].intersection(remaining))
            remaining.difference_update(to_remove)
        return current_set

    # Better Greedy:
    # 1. Calculate degrees.
    # 2. Pick min degree.
    # 3. Remove.
    # Update degrees?
    # For 5280 nodes, simple O(N^2) greedy might be too slow if repeated.
    # But we only do it once.
    
    # Let's do a simple random greedy first, then SA.
    
    S = set()
    # Random start
    # Pick 50 random nodes, check validity?
    # No, just start with empty set and add random non-conflicting nodes.
    
    # Let's use the random greedy from above but with a time limit or fixed steps.
    # Actually, let's just use the random greedy.
    
    S = greedy_is()
    best_set = S.copy()
    best_size = len(S)
    
    # SA
    T = 1.0
    cooling = 0.999
    # 50,000 steps
    for step in range(50000):
        v = random.randint(0, total_nodes - 1)
        
        if v in S:
            continue
            
        N = adj[v].intersection(S)
        
        if len(N) == 0:
            S.add(v)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
        else:
            delta = 1 - len(N)
            if delta > 0:
                S.difference_update(N)
                S.add(v)
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()
            else:
                if delta == 0:
                    if random.random() < 0.5:
                        S.difference_update(N)
                        S.add(v)
                else:
                    p = np.exp(delta / T)
                    if random.random() < p:
                        S.difference_update(N)
                        S.add(v)
                        
        T *= cooling
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # Construct vectors
    vectors = []
    for node_idx in best_set:
        sup_idx = node_idx // 16
        sign_idx = node_idx % 16
        sup = supports[sup_idx]
        signs = sign_tuples[sign_idx]
        v = [0] * D
        for k, s in enumerate(sup):
            v[s] = signs[k]
        vectors.append(v)
        
    return np.array(vectors, dtype=np.int64)
