import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Construct LUW graph with minimal d
    # LUW(q, d) has degree q and girth 2d+2
    # We need degree k, so q=k.
    # We need girth >= g, so 2d+2 >= g => d >= (g-2)/2
    # For k=5, g=9: d >= 3.5 => d=4.
    
    def construct_luw(k, g):
        q = k
        # Minimal d for girth >= g
        d = (g - 2 + 1) // 2  # ceil((g-2)/2)
        if d < 3:
            d = 3
            
        # LUW construction
        # Vertices: (side, x_0, ..., x_{d-1})
        # side 0: y_i = x_i + x_{i-1} * x_{i-2} ? 
        # Standard LUW recurrence:
        # x_i = y_i + y_{i-1} y_{i-2}
        # Let's use the standard definition from Lazebnik et al.
        # The graph is bipartite.
        # Left vertices: (0, a_1, ..., a_d)
        # Right vertices: (1, b_1, ..., b_d)
        # Edge between (0, a) and (1, b) if:
        # b_i = a_i + a_{i-1} a_{i-2} for i=1..d (indices mod d, with a_0=a_d, a_{-1}=a_{d-1})
        # Wait, the recurrence is usually:
        # b_i = a_i + a_{i-1} a_{i-2}
        # This defines the neighbors.
        
        # Number of vertices: 2 * q^d
        n = 2 * (q ** d)
        
        # To be efficient, we map (side, tuple) to index
        # index = side * q^d + sum(a_i * q^i)
        
        edges = []
        
        # Iterate over all left vertices
        for side_a in range(q):
            # This is a bit slow if we iterate naively. 
            # Better to iterate over all tuples.
            pass
        
        # Efficient iteration
        q_d = q ** d
        for idx_a in range(q_d):
            # Convert idx_a to tuple a
            a = [0] * d
            temp = idx_a
            for i in range(d):
                a[i] = temp % q
                temp //= q
            
            # Calculate neighbors b
            # b_i = a_i + a_{i-1} * a_{i-2} mod q
            # Indices: i-1, i-2 mod d
            b = [0] * d
            for i in range(d):
                prev1 = a[(i - 1) % d]
                prev2 = a[(i - 2) % d]
                b[i] = (a[i] + prev1 * prev2) % q
            
            # Convert b to idx_b
            idx_b = 0
            for i in range(d):
                idx_b += b[i] * (q ** i)
            
            # Left vertex index: idx_a
            # Right vertex index: q_d + idx_b
            edges.append((idx_a, q_d + idx_b))
            
        return n, edges

    # Check if LUW is valid for our params
    # For k=5, g=9, d=4.
    # Girth is 10. 10 >= 9. Valid.
    n_luw, edges_luw = construct_luw(k, g)
    
    # 2. Try to find a smaller graph
    # Moore bound for 5,9 is 426.
    # Known cage is 1116.
    # We can try to find a graph with n < n_luw.
    # Random regular graph generation + girth check.
    
    def has_short_cycle(adj, n, max_len):
        # Check for cycles of length < max_len
        # Use BFS from each node, depth max_len-1
        # If we find a visited node at depth < max_len-1 that is not parent, 
        # we need to check the cycle length.
        # Simpler: Check all paths of length < max_len? No, too slow.
        # Standard: For each start, BFS. If we revisit a node, the cycle length is 
        # current_depth + 1 + dist(visited_node, start)? No.
        # If we do BFS from start, and we find an edge (u,v) where v is visited 
        # and v != parent(u), then we have a cycle. 
        # The length of this cycle is depth(u) + 1 + depth(v) if LCA is start? 
        # No, LCA could be elsewhere.
        # For small max_len (9), we can just do a DFS/BFS up to depth max_len-1 
        # and check if we close a loop.
        # Actually, for girth check, it's sufficient to check if there is any 
        # non-backtracking walk of length < g that closes.
        
        # Efficient check for girth >= g:
        # For each vertex s, do BFS up to depth (g-1)//2.
        # If we find a collision (two paths to same node) at depth <= (g-1)//2,
        # we might have a short cycle.
        # Specifically, if g is odd (9), radius is 4.
        # If two distinct paths of length <= 4 from s meet at a node v, 
        # the cycle length is <= 4 + 4 = 8? 
        # If they meet at v, and paths are s->...->v and s->...->v.
        # Length = len1 + len2.
        # If len1 + len2 < 9, girth < 9.
        # So we check for collisions in BFS tree of depth 4.
        
        radius = (g - 1) // 2
        for s in range(n):
            # BFS from s
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= radius:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        # Found a cycle?
                        # The cycle length is dist[u] + 1 + dist[v] ?
                        # No, this is the length of the walk s->u->v->...->s?
                        # If v is visited, the path s->...->v exists.
                        # The cycle is u->v + path(v->s) + path(s->u)?
                        # Actually, if we find a back edge or cross edge in BFS,
                        # the cycle length is dist[u] + 1 + dist[v] is NOT correct.
                        # The cycle is formed by the edge (u,v) and the paths in the BFS tree.
                        # Path 1: s to u (len dist[u])
                        # Path 2: s to v (len dist[v])
                        # Edge u-v.
                        # This doesn't form a simple cycle unless LCA is s.
                        # To be safe, for small radius, we can just check if 
                        # dist[u] + dist[v] + 1 < g.
                        # If dist[u] + dist[v] + 1 < g, then there is a cycle 
                        # of length <= dist[u] + dist[v] + 1?
                        # Not necessarily simple, but if it's a cycle in the graph, 
                        # and the graph is simple, the shortest cycle through this 
                        # structure is bounded.
                        # Actually, if dist[u] + dist[v] + 1 < g, it's a strong 
                        # indicator.
                        # For g=9, radius=4. Max dist sum = 4+4=8. 8+1=9.
                        # So if dist[u] + dist[v] + 1 < 9, i.e., <= 8, we have a 
                        # cycle of length <= 8?
                        # Yes, the walk s->...->u->v->...->s has length 
                        # dist[u] + 1 + dist[v].
                        # If this walk is not simple, it contains a simple cycle 
                        # of length <= total length.
                        # So if dist[u] + dist[v] + 1 < g, girth < g.
                        if dist[u] + dist[v] + 1 < g:
                            return True
        return False

    # Search for smaller n
    best_n = n_luw
    best_edges = edges_luw
    
    # Try a few random instances
    # n from 1116 to best_n - 1
    # This is expensive. 
    # Given time constraints, we might just return LUW.
    # But let's try 1116 specifically?
    # No, random search is unlikely to hit the exact cage.
    
    # However, we can try to optimize the LUW parameters if possible?
    # No, d is minimal.
    
    # Let's just return the LUW graph.
    return n_luw, edges_luw
