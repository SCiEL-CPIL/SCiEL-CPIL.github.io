import itertools
import numpy as np

def construct():
    D = 11
    weight = 5
    
    # Generate all supports of size 'weight'
    supports = list(itertools.combinations(range(D), weight))
    n_sup = len(supports)
    
    # Map support to index
    sup_to_idx = {s: i for i, s in enumerate(supports)}
    
    # Generate vectors for each support
    # We use even parity sign patterns to ensure min distance 2 within support
    # v is a tuple of (support_idx, sign_tuple)
    # sign_tuple is for the positions in the support
    vectors = []
    vec_to_sup = []
    vec_to_signs = []
    
    for s in supports:
        # Generate all 2^weight sign patterns
        # Keep only those with even number of -1s (or even sum of signs? No, parity of indices? No, parity of -1s)
        # Actually, condition was d_H >= 2.
        # Even parity code (sum of signs is even? No, number of -1s is even) has min distance 2.
        for signs in itertools.product([1, -1], repeat=weight):
            if sum(1 for x in signs if x == -1) % 2 == 0:
                vectors.append(tuple(signs))
                vec_to_sup.append(s)
                vec_to_signs.append(signs)
                
    n_vec = len(vectors)
    
    # Build conflict graph
    # adj[i] is a set of j such that vec i and vec j conflict
    adj = [set() for _ in range(n_vec)]
    
    # Group vector indices by support
    sup_to_vecs = {s: [] for s in supports}
    for i, s in enumerate(vec_to_sup):
        sup_to_vecs[s].append(i)
        
    # Iterate over pairs of supports with intersection 4
    # To do this efficiently, we can just iterate over all pairs of supports?
    # n_sup = 462. Pairs = 462*461/2 = 106k.
    # For each pair, check intersection size.
    # If 4, check vector conflicts.
    
    for i in range(n_sup):
        s_i = supports[i]
        vecs_i = sup_to_vecs[s_i]
        for j in range(i+1, n_sup):
            s_j = supports[j]
            # Check intersection
            inter = set(s_i).intersection(s_j)
            if len(inter) == 4:
                vecs_j = sup_to_vecs[s_j]
                # For each pair of vectors, check dot product
                # v_i dot v_j
                # v_i has signs on s_i. v_j has signs on s_j.
                # Dot product is sum over inter of v_i[x] * v_j[x].
                # We need dot product <= 3.
                # If dot product == 4, conflict.
                for vi_idx in vecs_i:
                    signs_i = vec_to_signs[vi_idx]
                    # Map signs to positions in s_i
                    # s_i is a tuple. inter is a set.
                    # We need to extract signs for positions in inter.
                    # This is slow if done naively.
                    
                    # Optimization: Precompute sign vectors as dicts or arrays?
                    # Or just compute dot product.
                    pass
                    
    # This nested loop might be slow in Python if not careful.
    # 106k pairs * 16*16 vectors = 27M dot product calcs.
    # Each dot product is sum of 4 products.
    # 27M * 4 = 100M ops. Might be too slow in pure Python (120s limit).
    
    # Better approach:
    # For each pair of supports with intersection 4, we can determine conflicts analytically.
    # v_i and v_j conflict if they match on all 4 intersection positions.
    # Let I be the intersection (size 4).
    # v_i is determined by signs on I and 1 unique pos.
    # v_j is determined by signs on I and 1 unique pos.
    # Conflict if signs on I are identical.
    # For a fixed v_i, how many v_j match on I?
    # v_i has a specific pattern on I.
    # v_j must have this pattern on I.
    # v_j also has even parity on S_j.
    # Pattern on I has parity p.
    # v_j unique pos must have parity p to make total even?
    # Sum of signs on I + sign on unique = 0 (mod 2)?
    # No, "even number of -1s".
    # Let k be number of -1s in pattern on I.
    # Total -1s in v_j = k + (1 if unique is -1 else 0).
    # We need this to be even.
    # So unique sign is determined by k.
    # So for each v_i, there is exactly 1 v_j that matches on I and has even parity.
    # So for each v_i, there is 1 conflict in V_j.
    # So we add 1 edge for each v_i in V_i to the specific v_j in V_j.
    # And symmetrically.
    
    # So we just need to find the matching vector.
    
    # Let's re-implement the conflict building using this logic.
    
    adj = [set() for _ in range(n_vec)]
    
    # Precompute a map for each support: sign_pattern_on_support -> vector_index
    # But we only care about patterns on the intersection.
    # Let's just iterate.
    
    for i in range(n_sup):
        s_i = supports[i]
        vecs_i = sup_to_vecs[s_i]
        for j in range(i+1, n_sup):
            s_j = supports[j]
            inter = set(s_i).intersection(s_j)
            if len(inter) == 4:
                # Find the unique elements
                u_i = [x for x in s_i if x not in inter][0]
                u_j = [x for x in s_j if x not in inter][0]
                
                # For each vector in vecs_i, find the conflicting vector in vecs_j
                for vi_idx in vecs_i:
                    signs_i = vec_to_signs[vi_idx]
                    # Extract signs on inter
                    # s_i is a tuple. inter is a set.
                    # We need to map positions in inter to signs.
                    # This is still a bit messy.
                    pass

    # This is getting complicated to optimize in pure Python within the edit constraints.
    # However, the current program is already slow-ish (greedy on 330 nodes).
    # If I just switch to Weight 5 and use a simpler heuristic (e.g., just take all even parity vectors, then remove conflicts greedily), it might work.
    
    # Let's try a different strategy:
    # 1. Start with all 7392 vectors.
    # 2. Identify all conflicting pairs.
    # 3. While conflicts exist, remove one vector from each conflicting pair (greedy).
    
    # To identify conflicts, we can use the analytical insight:
    # For each pair of supports with intersection 4, there are 16 conflicting pairs (one for each vector in S_i).
    # Total conflicts = 6930 * 16 = 110,880 edges.
    # We can build the adjacency list by iterating support pairs.
    
    # To make it fast, we need to quickly find the conflicting vector index.
    # Let's create a dictionary for each support:
    # key: tuple of signs on the 4 intersection positions? No, intersection changes.
    
    # Alternative:
    # Just build the full graph using numpy broadcasting?
    # 7392 vectors.
    # Represent each vector as an 11-dim array of 0, 1, -1.
    # 7392 x 11 matrix.
    # Compute Gram matrix? 7392 x 7392.
    # 54M floats. 400MB memory. Too much?
    # But we can do it in chunks.
    
    # Or, since we know the structure, we can just generate the edges.
    
    # Let's go back to the support pair iteration.
    # We need to map (support, signs_on_support) to vector_index.
    # This is easy: vec_to_sup and vec_to_signs.
    # But signs are on the support positions.
    
    # Let's create a dict: sup -> {signs_tuple: vec_idx}
    sup_signs_to_vec = {}
    for s in supports:
        d = {}
        for idx in sup_to_vecs[s]:
            d[vec_to_signs[idx]] = idx
        sup_signs_to_vec[s] = d
        
    # Now, for each pair of supports with intersection 4:
    # We need to find the vector in s_j that matches s_i on the intersection.
    # This requires checking 16 vectors in s_j for each vector in s_i.
    # 16*16 = 256 checks per support pair.
    # 6930 pairs * 256 = 1.7M checks.
    # This is fast enough in Python.
    
    adj = [set() for _ in range(n_vec)]
    
    for i in range(n_sup):
        s_i = supports[i]
        vecs_i = sup_to_vecs[s_i]
        for j in range(i+1, n_sup):
            s_j = supports[j]
            inter = set(s_i).intersection(s_j)
            if len(inter) == 4:
                vecs_j = sup_to_vecs[s_j]
                # For each vi in vecs_i
                for vi_idx in vecs_i:
                    signs_i = vec_to_signs[vi_idx]
                    # We need to find vj in vecs_j such that signs match on inter.
                    # We can just loop vecs_j.
                    for vj_idx in vecs_j:
                        signs_j = vec_to_signs[vj_idx]
                        # Check if they match on inter
                        # This requires mapping signs to positions.
                        # s_i is a tuple of positions.
                        # signs_i is a tuple of signs corresponding to s_i.
                        # We need to compare signs for positions in inter.
                        # This is O(4) per check.
                        # 1.7M * 4 = 7M ops. Fast.
                        
                        match = True
                        for pos in inter:
                            idx_i = s_i.index(pos)
                            idx_j = s_j.index(pos)
                            if signs_i[idx_i] != signs_j[idx_j]:
                                match = False
                                break
                        if match:
                            adj[vi_idx].add(vj_idx)
                            adj[vj_idx].add(vi_idx)
                            
    # Now run Greedy MIS on adj
    remaining = set(range(n_vec))
    independent_set = []
    
    while remaining:
        # Find min degree
        min_deg = float('inf')
        best_v = None
        for v in remaining:
            deg = len(adj[v].intersection(remaining))
            if deg < min_deg:
                min_deg = deg
                best_v = v
                if deg == 0: break
        independent_set.append(best_v)
        to_remove = {best_v}.union(adj[best_v].intersection(remaining))
        remaining.difference_update(to_remove)
        
    # Construct final vectors
    result = []
    for idx in independent_set:
        s = vec_to_sup[idx]
        signs = vec_to_signs[idx]
        v = [0] * D
        for k, pos in enumerate(s):
            v[pos] = signs[k]
        result.append(v)
        
    return np.array(result, dtype=np.int64)
