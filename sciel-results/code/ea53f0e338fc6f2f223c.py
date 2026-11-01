"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    # We try to find the smallest d such that the girth is at least g.
    # Theoretical girth is d + 5.
    # For g=9, d=4 gives theoretical girth 9.
    # We verify the girth to be safe.
    
    q = k
    
    def build_luw(d):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        # Build adjacency list for girth check
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        return n, edges, adj

    def check_girth(n, adj, min_girth):
        # Check for cycles of length < min_girth
        # Since graph is bipartite, check even lengths 4, 6, ...
        # BFS from each vertex
        for start in range(n):
            # dist[v] = shortest distance from start
            # prev[v] = previous vertex in shortest path
            dist = [-1] * n
            prev = [-1] * n
            dist[start] = 0
            queue = [start]
            
            # We only need to check up to depth min_girth // 2 - 1?
            # Actually, if we find a cycle of length L, it means 
            # dist[u] + dist[v] + 1 = L for an edge (u,v) where u,v are in the BFS tree?
            # Simpler: Just run BFS and check for edges that connect nodes at depth d1, d2
            # such that d1 + d2 + 1 < min_girth.
            
            # Standard BFS girth check:
            # If an edge (u, v) exists and dist[u] + dist[v] + 1 < min_girth, 
            # AND u != start, v != start, and one is not parent of the other?
            # Actually, if dist[u] + dist[v] + 1 < min_girth, there is a cycle of that length.
            
            queue_idx = 0
            while queue_idx < len(queue):
                u = queue[queue_idx]
                queue_idx += 1
                
                if dist[u] >= min_girth // 2:
                    continue # No need to go deeper if we are looking for small cycles
                
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        prev[v] = u
                        queue.append(v)
                    else:
                        # Found a back edge or cross edge
                        # Cycle length = dist[u] + dist[v] + 1
                        # But this might be the tree edge itself (v is parent of u)
                        if v == prev[u]:
                            continue
                        # Also skip if u is parent of v (handled by dist check? No, dist[v] = dist[u]+1)
                        # If dist[v] == dist[u] + 1, v is child.
                        # If dist[u] == dist[v] + 1, u is child.
                        # The cycle formed is dist[u] + dist[v] + 1.
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
            # Reset dist/prev for next start? 
            # Doing this for every start is O(N^2). 
            # For N=1250, 1250*1250 = 1.5M operations. Fast.
        
        return True

    # Try d = g - 5 first (optimistic)
    d_opt = g - 5
    if d_opt < 3: d_opt = 3
    
    start_time = time.time()
    
    # Try optimistic d
    n_opt, edges_opt, adj_opt = build_luw(d_opt)
    
    # Verify girth
    if check_girth(n_opt, adj_opt, g):
        return n_opt, edges_opt
    else:
        # Fallback to conservative d
        d_con = g - 4
        if d_con < d_opt: d_con = d_opt + 1 # Ensure larger
        # Actually, if d_opt failed, try d_opt + 1 or g-4
        d_con = max(d_opt + 1, 5) # For g=9, d=5
        
        n_con, edges_con, _ = build_luw(d_con)
        return n_con, edges_con

# EVOLVE-BLOCK-END
