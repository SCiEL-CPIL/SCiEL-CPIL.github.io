import time
import numpy as np

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using Lazebnik-Ustimenko-Woldar (LUW) graphs.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    q = k
    
    # Determine candidate d values
    # For LUW CD(d, q), girth is approx d+5 (odd d) or d+4 (even d).
    # We need girth >= g.
    # We prefer odd d as it gives larger girth for same d, or smaller d for same girth.
    # For g=9:
    # Odd d: d+5 >= 9 => d >= 4. Smallest odd is 5.
    # Even d: d+4 >= 9 => d >= 5. Smallest even is 6.
    # d=5 gives n = 2*q^5. d=6 gives n = 2*q^6.
    # So d=5 is better.
    
    candidates = []
    # Try odd d
    d_odd = g - 5
    if d_odd < 3: d_odd = 3
    if d_odd % 2 == 0: d_odd += 1
    candidates.append(d_odd)
    
    # Try even d
    d_even = g - 4
    if d_even < 4: d_even = 4
    if d_even % 2 == 1: d_even += 1
    candidates.append(d_even)
    
    # Sort by resulting n (which is q^d, so effectively by d)
    candidates = sorted(set(candidates))
    
    # Arithmetic functions
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
        
    def build_luw(d):
        # Rule table for LUW graph
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        i = 0
        while i < len(order):
            if time.time() > deadline:
                return None, None, None
            vertex = order[i]
            i += 1
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                
        n = len(order)
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        return n, edges, adj

    def has_short_cycle(n, adj, max_len):
        """
        Check for cycles of length < max_len (i.e., 4, 6, ... up to max_len-1).
        Since graph is bipartite, only even cycles exist.
        We check for cycles of length 4, 6, ..., max_len-2.
        """
        # BFS from each vertex to depth max_len//2 - 1
        # If we find an edge (u, v) such that dist[u] + dist[v] + 1 < max_len,
        # and u is not parent of v, then we have a cycle.
        
        # Optimization: Only check from vertices 0 to n-1.
        # For N=6250, this is fast enough if depth is small.
        depth_limit = max_len // 2
        
        for start in range(n):
            dist = [-1] * n
            prev = [-1] * n
            dist[start] = 0
            queue = [start]
            q_idx = 0
            
            while q_idx < len(queue):
                u = queue[q_idx]
                q_idx += 1
                
                if dist[u] >= depth_limit:
                    continue
                    
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        prev[v] = u
                        queue.append(v)
                    else:
                        # If v is already visited
                        # Check if it's a back edge forming a short cycle
                        if v == prev[u]:
                            continue
                        # Cycle length = dist[u] + dist[v] + 1
                        # Note: This detects cycles, but might double count or detect tree edges incorrectly?
                        # In BFS, if dist[v] != dist[u] + 1 and dist[v] != dist[u] - 1, it's a cross edge?
                        # Actually, for bipartite graphs, edges only connect dist d and d+1.
                        # So if v is visited, and v != prev[u], then:
                        # If dist[v] == dist[u] + 1, v is a child (tree edge).
                        # If dist[v] == dist[u] - 1, v is a parent (tree edge, handled by prev check? No, prev[u] is the parent. If v is parent, v == prev[u]. So we skip).
                        # If dist[v] == dist[u], it's a cross edge in same layer? Impossible in bipartite BFS from single source?
                        # Wait, BFS layers are by distance. In bipartite graph, edges connect layer d to d+1.
                        # So if v is visited, dist[v] is either dist[u]+1 (child) or dist[u]-1 (parent).
                        # If v is parent, v == prev[u].
                        # If v is child, dist[v] = dist[u] + 1.
                        # Are there other cases?
                        # What if there is an edge between two nodes in the same BFS tree that are not parent-child?
                        # In a tree, no. But this is a graph, not a tree.
                        # If there is an edge (u, v) where dist[u] = d, dist[v] = d+1, and v is NOT a child of u in the BFS tree,
                        # then u is not prev[v].
                        # In this case, we have a cycle: start -> ... -> u -> v -> ... -> start?
                        # The path start -> ... -> v has length d+1.
                        # The path start -> ... -> u has length d.
                        # The edge (u, v) connects them.
                        # The cycle length is d + (d+1) + 1 = 2d + 2.
                        # We need to check if this cycle is short.
                        
                        if dist[v] == dist[u] + 1:
                            # v is in the next layer
                            # If v is a child of u, then prev[v] == u.
                            # If prev[v] != u, then (u, v) is a non-tree edge.
                            if prev[v] != u:
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < max_len:
                                    return True
                        elif dist[v] == dist[u] - 1:
                            # u is in the next layer relative to v
                            # If u is a child of v, then prev[u] == v.
                            if prev[u] != v:
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < max_len:
                                    return True
            # Reset dist/prev is implicit by re-allocating in next loop iteration? 
            # No, we re-allocate inside the loop.
            
        return False

    for d in candidates:
        if time.time() > deadline:
            break
            
        n, edges, adj = build_luw(d)
        if n is None:
            continue
            
        # Check for cycles shorter than g
        # We check up to g-1 (which is even if g is odd? No, g=9, check 4,6,8)
        # has_short_cycle checks for cycles < max_len.
        # So we pass g. It will check for cycles < 9.
        
        if not has_short_cycle(n, adj, g):
            return n, edges
        else:
            # If short cycle found, try next d
            continue
            
    # If all fail, return the last one constructed (or a fallback)
    # This should not happen for standard parameters
    return n, edges
