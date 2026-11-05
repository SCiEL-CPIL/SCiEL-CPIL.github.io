import time
import itertools
import numpy as np
import random

def construct():
    start_time = time.time()
    DEADLINE = 115
    
    n = 16
    
    def pair_to_bit(i, j):
        if i > j: i, j = j, i
        return 7*i - i*(i-1)//2 + (j - i - 1)

    base_q3_edges = set()
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                base_q3_edges.add((i, j))
    
    unique_q3_masks = set()
    for p in itertools.permutations(range(8)):
        mask = 0
        for u, v in base_q3_edges:
            pu = p[u]
            pv = p[v]
            if pu > pv: pu, pv = pv, pu
            bit = pair_to_bit(pu, pv)
            mask |= (1 << bit)
        unique_q3_masks.add(mask)
    
    q3_masks = list(unique_q3_masks)
    
    # Precompute all 6-combinations of 16 vertices
    all_6_combos = list(itertools.combinations(range(n), 6))
    
    def check_q3_in_subset(adj, subset):
        # subset is tuple of 8
        # Check degrees of first two? No, u,v are not necessarily first two in subset.
        # We pass u, v separately.
        pass

    def run_trial(seed):
        random.seed(seed)
        adj = np.zeros((n, n), dtype=bool)
        edges = []
        
        candidate_edges = list(itertools.combinations(range(n), 2))
        random.shuffle(candidate_edges)
        
        for u, v in candidate_edges:
            if adj[u, v]:
                continue
            
            adj[u, v] = True
            adj[v, u] = True
            
            # Check for Q3 containing u, v
            # Filter 6-combos that don't contain u or v
            found_q3 = False
            for six in all_6_combos:
                # Check if u or v in six
                if u in six or v in six:
                    continue
                
                # Degree prune
                # u's neighbors in S (excluding v)
                # S = {u, v} + six
                # Neighbors of u in six
                deg_u = 0
                deg_v = 0
                for x in six:
                    if adj[u, x]: deg_u += 1
                    if adj[v, x]: deg_v += 1
                
                if deg_u < 2 or deg_v < 2: # In Q3, u has 2 other neighbors, v has 2 other neighbors
                    continue
                
                # Build mask
                subset = (u, v) + six
                # We need to map subset to 0..7 for pair_to_bit?
                # pair_to_bit expects indices 0..7.
                # We can just use the vertex IDs 0..15 in a 16x16 bitmask?
                # No, q3_masks are for 0..7.
                # We must map subset vertices to 0..7.
                # This mapping is expensive.
                
                # Alternative: Precompute q3_masks for ALL 8-subsets of 16 vertices?
                # C(16,8) = 12870 subsets.
                # For each subset, there are 336 Q3 patterns.
                # 12870 * 336 = 4.3M masks.
                # Memory: 4.3M * 4 bytes = 17MB.
                # Precomputation time: 4.3M * (cost to generate mask).
                # This might be too slow to start.
                
                # Let's stick to mapping.
                # To speed up mapping, we can use a dictionary or list.
                # But doing it 3000 times per edge is slow.
                
                # Let's try a different approach for Q3 check.
                # Instead of bitmask of K8, use the adjacency of the 8 vertices directly.
                # But we still need to check against 336 patterns.
                
                # Let's optimize the mapping.
                # Create a local list of 8 vertices.
                # Map each vertex to its index in the list.
                
                # Actually, we can precompute the "Q3 edge lists" for each specific set of 8 vertices?
                # No, that's what we are trying to avoid.
                
                # Let's accept the mapping cost.
                # 3000 subsets * 8 log 8 mapping = ~24000 ops.
                # Plus 336 bit checks.
                # Total ~25000 ops per subset? No, 336 is small.
                # Mapping is the bottleneck.
                
                # Can we avoid mapping?
                # Represent Q3 masks as sets of pairs of vertex IDs?
                # Then for a given subset, we just check if all pairs in Q3_pattern are in adj?
                # 336 patterns * 12 edges = 4000 checks.
                # This is similar to bitmask but without mapping.
                # But 4000 boolean checks might be slower than bit ops.
                
                # Let's try to optimize the mapping.
                # We can precompute the index of each vertex in the subset?
                # No, subset changes.
                
                # Let's try to reduce the number of subsets checked.
                # The degree prune `deg_u < 2` is good.
                
                # Let's proceed with the current approach but ensure it's as fast as possible.
                
                # Optimization: Precompute `six` list as a tuple for faster iteration?
                # `all_6_combos` are tuples.
                
                # Let's just run it and see if it fits in time.
                
                # To make it faster, we can precompute the 336 Q3 edge lists as lists of pairs (i,j) where i,j in 0..7.
                # Then for a subset, we map vertices to 0..7.
                # Then for each Q3 edge list, check if adj[subset[i], subset[j]] is True for all 12 edges.
                # 336 * 12 = 4032 checks.
                # This avoids bit manipulation and mask building.
                
                # Let's use this.
                pass

        return edges

    # Re-implementing with direct edge checks
    q3_edge_lists = []
    for p in itertools.permutations(range(8)):
        edges_in_perm = []
        for u, v in base_q3_edges:
            pu = p[u]
            pv = p[v]
            edges_in_perm.append((pu, pv))
        # Deduplicate edge lists
        # Sort edges in each list, then sort the list of edges
        edges_in_perm.sort()
        key = tuple(edges_in_perm)
        if key not in q3_edge_lists:
            q3_edge_lists.append(key)
            
    # q3_edge_lists now contains 336 tuples of 12 pairs.
    
    def run_trial(seed):
        random.seed(seed)
        adj = np.zeros((n, n), dtype=bool)
        edges = []
        
        candidate_edges = list(itertools.combinations(range(n), 2))
        random.shuffle(candidate_edges)
        
        for u, v in candidate_edges:
            if adj[u, v]:
                continue
            
            adj[u, v] = True
            adj[v, u] = True
            
            found_q3 = False
            for six in all_6_combos:
                if u in six or v in six:
                    continue
                
                # Degree prune
                deg_u = 0
                deg_v = 0
                for x in six:
                    if adj[u, x]: deg_u += 1
                    if adj[v, x]: deg_v += 1
                if deg_u < 2 or deg_v < 2:
                    continue
                
                # Check Q3
                subset = (u, v) + six
                # Map subset to 0..7
                # We can just use the vertex IDs directly if we store Q3 patterns as sets of vertex pairs from 0..15?
                # No, Q3 patterns are relative.
                
                # We need to check if there exists a bijection f: {0..7} -> subset such that
                # for all (i,j) in Q3_pattern, adj[subset[f(i)], subset[f(j)]] is True.
                # This is the subgraph isomorphism problem for Q3.
                # But we precomputed the 336 patterns for 0..7.
                # So we need to map subset to 0..7.
                
                # Let's create a list `s_list` = list(subset)
                # And for each q3_pattern in q3_edge_lists:
                #   Check if adj[s_list[i], s_list[j]] for all (i,j) in q3_pattern.
                
                s_list = list(subset)
                for pattern in q3_edge_lists:
                    ok = True
                    for i, j in pattern:
                        if not adj[s_list[i], s_list[j]]:
                            ok = False
                            break
                    if ok:
                        found_q3 = True
                        break
                if found_q3:
                    break
            
            if found_q3:
                adj[u, v] = False
                adj[v, u] = False
            else:
                edges.append([u, v])
                
        return edges

    best_edge_count = 0
    best_edges = []
    trial = 0
    while time.time() < DEADLINE:
        trial += 1
        edges = run_trial(trial)
        if len(edges) > best_edge_count:
            best_edge_count = len(edges)
            best_edges = edges

    return best_edges
