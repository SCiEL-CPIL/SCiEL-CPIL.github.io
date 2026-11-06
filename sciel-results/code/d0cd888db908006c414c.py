import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145

    n = 16
    # Initial graph: K_3,13 with K_3 on 0,1,2
    edges = []
    for a in range(3):
        for b in range(3, 16):
            edges.append([a, b])
    edges.append([0, 1])
    edges.append([0, 2])
    edges.append([1, 2])
    
    # Build adjacency list/matrix
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute Q3 structure
    # Vertices 0..7
    # Edges:
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

    # Precompute all 8-subsets containing a specific pair?
    # No, we can generate them on the fly or precompute all 8-subsets and filter.
    # Precomputing all 8-subsets: 12870.
    # For each subset, store the tuple.
    all_8_subsets = list(itertools.combinations(range(n), 8))
    
    # Map edge (u,v) to list of subsets containing it
    # This is expensive to build fully, but we can do it lazily or precompute.
    # 12870 subsets * 8 vertices = ~100k ops.
    # Let's precompute `subsets_containing_edge`
    subsets_containing_edge = { (u, v): [] for u in range(n) for v in range(n) if u < v }
    for s in all_8_subsets:
        for i in range(8):
            for j in range(i+1, 8):
                u, v = s[i], s[j]
                if u > v: u, v = v, u
                subsets_containing_edge[(u, v)].append(s)

    def has_q3_in_subset(subset, adj):
        # subset is a tuple of 8 vertex indices
        # Map subset indices 0..7 to actual vertex indices
        # Check degrees in induced subgraph
        deg = [0] * 8
        sub_adj = [0] * 8
        
        # Build induced subgraph adj masks relative to 0..7
        for i in range(8):
            u = subset[i]
            mask = 0
            # Iterate over other vertices in subset
            for j in range(8):
                if i == j: continue
                v = subset[j]
                if (adj[u] >> v) & 1:
                    mask |= (1 << j)
                    deg[i] += 1
            sub_adj[i] = mask
            
        # Filter: all vertices in Q3 have degree 3.
        # So all images must have degree >= 3 in sub_adj.
        for i in range(8):
            if deg[i] < 3:
                return False
                
        # DFS to find isomorphism
        # mapping[q_v] = sub_idx
        mapping = [-1] * 8
        used = 0
        
        # Order q_vertices by degree in remaining graph? 
        # For simplicity, fixed order 0..7. 
        # But we can optimize by picking the q_vertex with most constraints first?
        # All have degree 3.
        
        # We can use a bitmask for used vertices for speed
        used_mask = 0
        
        def dfs(q_v):
            nonlocal used_mask
            if q_v == 8:
                return True
            
            # Try each available sub_vertex
            # Optimization: check compatibility with already mapped neighbors
            # Get neighbors of q_v in Q3
            q_neighbors = q3_adj[q_v]
            
            # Candidates are unused vertices in sub_adj
            candidates = ~used_mask & 0xFF
            while candidates:
                lsb = candidates & -candidates
                s_v = lsb.bit_length() - 1
                candidates ^= lsb
                
                # Check if s_v is compatible with q_v
                ok = True
                # Check neighbors of q_v that are already mapped
                # Iterate through mapped neighbors of q_v
                # We can precompute mapped neighbors for q_v? 
                # Or just check all q_u
                for q_u in range(8):
                    if (q_neighbors >> q_u) & 1:
                        if mapping[q_u] != -1:
                            mapped_u = mapping[q_u]
                            # s_v must be adjacent to mapped_u in sub_adj
                            if not ((sub_adj[s_v] >> mapped_u) & 1):
                                ok = False
                                break
                
                if not ok:
                    continue
                
                # Also, check if s_v has enough potential neighbors for future mappings?
                # Skip for now.
                
                mapping[q_v] = s_v
                used_mask |= lsb
                if dfs(q_v + 1):
                    return True
                used_mask ^= lsb
                mapping[q_v] = -1
                
            return False
            
        return dfs(0)

    def has_q3_with_edge(u, v, adj):
        # Check only subsets containing u and v
        # u, v must be sorted
        if u > v: u, v = v, u
        for s in subsets_containing_edge[(u, v)]:
            if has_q3_in_subset(s, adj):
                return True
        return False

    # Greedy Add
    # List of all possible edges
    all_edges = []
    for u in range(n):
        for v in range(u+1, n):
            all_edges.append((u, v))
            
    current_edges_set = set()
    for e in edges:
        current_edges_set.add(tuple(e))
        
    missing_edges = [e for e in all_edges if e not in current_edges_set]
    
    # Shuffle to break symmetry?
    random.seed(42)
    random.shuffle(missing_edges)
    
    added = 0
    for u, v in missing_edges:
        if time.time() > deadline - 10:
            break
            
        # Tentatively add
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        current_edges_set.add((u, v))
        
        if has_q3_with_edge(u, v, adj):
            # Revert
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            current_edges_set.remove((u, v))
        else:
            added += 1
            
    # Convert back to list of lists
    result_edges = [list(e) for e in current_edges_set]
    
    return result_edges
