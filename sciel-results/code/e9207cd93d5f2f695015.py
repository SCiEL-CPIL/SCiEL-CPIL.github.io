import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145

    n = 16
    # Partition vertices into S1 (0-3) and S2 (4-15)
    S1 = list(range(4))
    S2 = list(range(4, 16))
    
    # Initial graph: K_4,12
    edges = []
    for u in S1:
        for v in S2:
            edges.append([u, v])
            
    # Build adjacency matrix (bitmask)
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute Q3 structure in K_4,4
    # Vertices 0..3 (A), 4..7 (B)
    # Edges are (i, j) for i in 0..3, j in 4..7
    # Q3 is K_4,4 minus a perfect matching
    # Perfect matchings are permutations of 0..3
    k44_q3_masks = []
    for perm in itertools.permutations(range(4)):
        # Remove edges (i, 4+perm[i])
        removed = set()
        for i in range(4):
            removed.add((i, 4 + perm[i]))
        
        q3_edges_local = []
        for i in range(4):
            for j in range(4):
                if (i, 4+j) not in removed:
                    q3_edges_local.append((i, 4+j))
        # Store as a set of tuples for easy lookup
        k44_q3_masks.append(frozenset(q3_edges_local))
        
    # Generate all Q3s in K_4,12
    # Each Q3 is defined by a 4-subset of S2 and a pattern from k44_q3_masks
    all_q3s = []
    # Map local indices 4..7 to global indices in S2
    # For a specific subset S2_sub of size 4, we map local 4->S2_sub[0], 5->S2_sub[1], etc.
    # Actually, the pattern assumes specific relative positions.
    # We just need to map the local edge (i, j) to global edge (S1[i], S2_sub[j-4])
    
    # Precompute subsets of S2
    s2_subsets = list(itertools.combinations(S2, 4))
    
    for s2_sub in s2_subsets:
        # s2_sub is a tuple of 4 global vertices from S2
        # Map local 4,5,6,7 to s2_sub[0], s2_sub[1], s2_sub[2], s2_sub[3]
        local_to_global_b = {4: s2_sub[0], 5: s2_sub[1], 6: s2_sub[2], 7: s2_sub[3]}
        
        for mask in k44_q3_masks:
            global_q3_edges = set()
            for u, v in mask:
                # u is 0..3 (S1), v is 4..7 (S2)
                gu = S1[u]
                gv = local_to_global_b[v]
                global_q3_edges.add((gu, gv))
            all_q3s.append(frozenset(global_q3_edges))
            
    # Greedy Hitting Set
    # active_q3s: list of indices of Q3s not yet hit
    active_q3s = list(range(len(all_q3s)))
    
    # edge_to_q3s: dict mapping edge to list of Q3 indices containing it
    # This allows fast update when an edge is removed
    edge_to_q3s = {}
    for idx, q3 in enumerate(all_q3s):
        for e in q3:
            if e not in edge_to_q3s:
                edge_to_q3s[e] = []
            edge_to_q3s[e].append(idx)
            
    # current_edges_set: set of edges currently in the graph
    current_edges_set = set(tuple(e) for e in edges)
    
    # We want to remove edges.
    # Keep a count of how many active Q3s each edge is in.
    edge_active_count = {e: len(q3s) for e, q3s in edge_to_q3s.items()}
    
    removed_edges = set()
    
    while active_q3s:
        if time.time() > deadline - 20:
            break
            
        # Find edge with max active count
        max_count = -1
        best_edge = None
        for e, cnt in edge_active_count.items():
            if cnt > max_count:
                max_count = cnt
                best_edge = e
                
        if best_edge is None:
            break
            
        # Remove best_edge
        current_edges_set.remove(best_edge)
        removed_edges.add(best_edge)
        
        # Update active Q3s
        # All Q3s containing best_edge are now hit
        q3s_to_remove = edge_to_q3s.get(best_edge, [])
        
        # We need to mark these Q3s as inactive and decrement counts for their other edges
        # To avoid modifying list while iterating, we process in batch
        # Create a set of Q3 indices to remove
        q3_indices_to_remove = set(q3s_to_remove)
        
        # Decrement counts for edges in these Q3s
        # We iterate through the Q3s being removed
        for idx in q3_indices_to_remove:
            q3_edges = all_q3s[idx]
            for e in q3_edges:
                if e == best_edge:
                    continue
                if e in edge_active_count:
                    edge_active_count[e] -= 1
                    
        # Remove Q3s from active list
        # active_q3s is a list. We can filter it.
        # Since active_q3s can be large, filtering is O(N).
        # But N is ~12k. 12k * 48 (max removals) is fine.
        active_q3s = [idx for idx in active_q3s if idx not in q3_indices_to_remove]
        
        # Also, we should remove best_edge from edge_to_q3s? 
        # Or just leave it, but it's no longer in the graph.
        # The loop condition checks active_q3s.
        
    # Now the graph is Q3-free.
    # Add Phase: Try to add back removed edges or non-edges of K_4,12
    # Non-edges of K_4,12 are:
    # 1. Edges within S1 (6 edges)
    # 2. Edges within S2 (66 edges)
    # Total 72 potential additions.
    # Plus the removed_edges (which were in K_4,12).
    
    # List of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Candidates to add: 
    # 1. Edges that were removed
    # 2. Edges that were never in K_4,12 (within S1 or within S2)
    
    candidates = []
    for e in all_edges:
        if e not in current_edges_set:
            candidates.append(e)
            
    random.seed(42)
    random.shuffle(candidates)
    
    # Precompute subsets containing edge for has_q3 check
    # Reuse logic from current program
    all_8_subsets = list(itertools.combinations(range(n), 8))
    subsets_containing_edge = { (u, v): [] for u in range(n) for v in range(n) if u < v }
    for s in all_8_subsets:
        for i in range(8):
            for j in range(i+1, 8):
                u, v = s[i], s[j]
                if u > v: u, v = v, u
                subsets_containing_edge[(u, v)].append(s)
                
    # Q3 adjacency for isomorphism check
    q3_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (4, 5), (4, 7),
        (3, 7),
        (6, 7)
    ]
    q3_adj = [0] * 8
    for u, v in q3_edges:
        q3_adj[u] |= (1 << v)
        q3_adj[v] |= (1 << u)

    def has_q3_in_subset(subset, adj):
        deg = [0] * 8
        sub_adj = [0] * 8
        for i in range(8):
            u = subset[i]
            mask = 0
            for j in range(8):
                if i == j: continue
                v = subset[j]
                if (adj[u] >> v) & 1:
                    mask |= (1 << j)
                    deg[i] += 1
            sub_adj[i] = mask
            if deg[i] < 3:
                return False
                
        mapping = [-1] * 8
        used_mask = 0
        
        def dfs(q_v):
            nonlocal used_mask
            if q_v == 8:
                return True
            candidates_mask = ~used_mask & 0xFF
            while candidates_mask:
                lsb = candidates_mask & -candidates_mask
                s_v = lsb.bit_length() - 1
                candidates_mask ^= lsb
                ok = True
                q_neighbors = q3_adj[q_v]
                for q_u in range(8):
                    if (q_neighbors >> q_u) & 1:
                        if mapping[q_u] != -1:
                            mapped_u = mapping[q_u]
                            if not ((sub_adj[s_v] >> mapped_u) & 1):
                                ok = False
                                break
                if not ok:
                    continue
                mapping[q_v] = s_v
                used_mask |= lsb
                if dfs(q_v + 1):
                    return True
                used_mask ^= lsb
                mapping[q_v] = -1
            return False
            
        return dfs(0)

    def has_q3_with_edge(u, v, adj):
        if u > v: u, v = v, u
        for s in subsets_containing_edge[(u, v)]:
            if has_q3_in_subset(s, adj):
                return True
        return False

    # Add edges
    for u, v in candidates:
        if time.time() > deadline - 5:
            break
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        current_edges_set.add((u, v))
        
        if has_q3_with_edge(u, v, adj):
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            current_edges_set.remove((u, v))
            
    result_edges = [list(e) for e in current_edges_set]
    return result_edges
