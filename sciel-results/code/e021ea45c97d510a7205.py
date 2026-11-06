"""Construct a C6-free graph on 40 vertices with many edges."""
import itertools

def construct():
    n = 40
    # Start with the base graph of 8 disjoint K5s (80 edges)
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj[u][v] = adj[v][u] = True
                edge_set.add((u, v) if u < v else (v, u))
    
    def has_c6(adj, u, v):
        """Check if adding edge (u, v) creates a C6."""
        # A C6 containing edge (u,v) has form u-a-b-c-d-v or v-a-b-c-d-u
        # We can check by looking for paths of length 4 between u and v in the graph
        # where the internal vertices don't include u or v.
        # Actually, more precisely: adding (u,v) creates a C6 iff there is a path of length 4
        # from u to v in the current graph G.
        # Path: u - x1 - x2 - x3 - x4 - v
        # All xi must be distinct and not u, v.
        
        # BFS from u to find all vertices at distance 4 from u
        # Then check if v is reachable via such paths where the last step is from a neighbor of v
        
        # More efficient: do BFS from u up to depth 4, tracking paths
        # But that's expensive. Let's use the structure.
        
        # Since our base graph is disjoint K5s, let's exploit that.
        # u and v are in different K5s (since we only add cross edges)
        # Let K_u be the K5 containing u, K_v be the K5 containing v.
        
        ku = u // 5
        kv = v // 5
        if ku == kv:
            return False  # Already in same component, no C6 created
        
        # For a C6 through (u,v): u - a - b - c - d - v
        # a must be in K_u (since u's neighbors are in K_u plus any cross-neighbors)
        # Wait, u may have cross-neighbors now.
        
        # General approach: 
        # Find all neighbors of u: N(u)
        # Find all neighbors of v: N(v)
        # We need a path of length 3 from some a in N(u) to some d in N(v)
        # such that a-b-c-d, all distinct, and a!=d, a!=c, b!=d, etc.
        # And the full path u-a-b-c-d-v has all 6 vertices distinct.
        
        Nu = [x for x in range(n) if adj[u][x]]
        Nv = [x for x in range(n) if adj[v][x]]
        
        # For each a in Nu and d in Nv, check if there's a path a-b-c-d of length 2
        # (i.e., b adjacent to both a and c, c adjacent to d, b!=c, b!=a, b!=d, c!=a, c!=d)
        # And also a!=d, and the path vertices don't include u or v.
        
        # This is still expensive. Let's use a different approach:
        # Check if there's a path of length 4 from u to v in G.
        # Use BFS with state (vertex, visited set) - but visited set makes it exponential.
        
        # Alternative: since n=40, we can do a limited BFS.
        # From u, expand level by level up to depth 4.
        # Level 0: {u}
        # Level 1: N(u)
        # Level 2: neighbors of level 1, excluding level 0 and 1
        # Level 3: neighbors of level 2, excluding levels 0-2
        # Level 4: neighbors of level 3, excluding levels 0-3
        # Check if v is in level 4.
        
        # But we also need the path to be simple (no repeated vertices), which BFS with
        # level tracking doesn't guarantee. However, for C6 detection, we need a specific
        # simple path. The BFS approach may have false positives.
        
        # Let's just do a direct check using the sparsity of cross-edges.
        # In practice, with few cross-edges, this should be fast.
        
        # Simpler: iterate over possible middle vertices b and c.
        # For each b, c:
        #   Check if u is adjacent to some a that is adjacent to b
        #   Check if v is adjacent to some d that is adjacent to c  
        #   Check if b is adjacent to c
        #   Check all 6 vertices distinct
        
        # Even simpler: for the specific case where most edges are within K5s,
        # the cross-edges are few. Let's try a practical approach.
        
        # Check: is there a path u-a-b-c-d-v?
        # Try all pairs (b, c) where adj[b][c]:
        #   Find a: neighbor of u and b, a != u, v, b, c
        #   Find d: neighbor of v and c, d != u, v, b, c, a
        
        for b in range(n):
            if b == u or b == v:
                continue
            for c in range(n):
                if c == u or c == v or c == b:
                    continue
                if not adj[b][c]:
                    continue
                # Find a in N(u) ∩ N(b), a not in {u, v, b, c}
                a_found = False
                d_found = False
                for a in range(n):
                    if a == u or a == v or a == b or a == c:
                        continue
                    if adj[u][a] and adj[a][b]:
                        a_found = a
                        break
                if not a_found:
                    continue
                for d in range(n):
                    if d == u or d == v or d == b or d == c or d == a_found:
                        continue
                    if adj[v][d] and adj[d][c]:
                        d_found = d
                        break
                if d_found:
                    return True
        return False
    
    # Try to add all possible cross edges that don't create a C6
    # Process in order of components
    added = 0
    for k1 in range(8):
        for k2 in range(k1 + 1, 8):
            for a in range(5):
                u = 5 * k1 + a
                for b in range(5):
                    v = 5 * k2 + b
                    if not adj[u][v] and not has_c6(adj, u, v):
                        adj[u][v] = adj[v][u] = True
                        edge_set.add((u, v) if u < v else (v, u))
                        added += 1
    
    edges = [list(e) for e in edge_set]
    return edges
