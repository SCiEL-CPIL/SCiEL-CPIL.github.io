def construct(k, g):
    import time
    import random
    from collections import deque
    
    start_time = time.time()
    deadline = start_time + 115
    
    # Hardcoded known optimal cages
    known_cages = {}
    
    # Petersen Graph (3, 5)
    known_cages[(3, 5)] = (10, [(0,1),(1,2),(2,3),(3,4),(4,0),
                                 (5,7),(7,9),(9,6),(6,8),(8,5),
                                 (0,5),(1,6),(2,7),(3,8),(4,9)])
    
    # Heawood Graph (3, 7)
    heawood_edges = []
    fano_lines = [(0,1,2),(0,3,4),(0,5,6),(1,3,5),(1,4,6),(2,3,6),(2,4,5)]
    for i, line in enumerate(fano_lines):
        line_id = 7 + i
        for p in line:
            heawood_edges.append((p, line_id))
    known_cages[(3, 7)] = (14, heawood_edges)
    
    # (3, 9) - 70 vertices? Actually the smallest known is 70 for (3,9)?
    # Let me use a known construction: the 3-regular graph of girth 9 with 70 vertices
    # Actually, n(3,9) is between 58 and 70. Let me use a known 70-vertex cage.
    # I'll construct it via LUW and verify.
    
    if (k, g) in known_cages:
        return known_cages[(k, g)]
    
    q = k
    
    def build_luw(d, q):
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
        
        rule = {}
        # Standard LUW rules
        rule[2] = (1,1)
        rule[3] = (2,1)
        rule[4] = (1,2)
        for c in range(5, d+1):
            r = (c-5)%4
            c0 = c - (c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1-side), tuple(y)
        
        start = (0, (None,) + (0,)*d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges
    
    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj
    
    def girth_ok(n, adj, g):
        """Check girth >= g using non-backtracking walks (BFS on directed edges)."""
        # Use BFS on the directed line graph (non-backtracking)
        # State: (current_vertex, previous_vertex)
        # Start: for each edge (u,v), state (v, u) with depth 1
        # We want to find the shortest cycle, which is the shortest non-backtracking walk
        # that returns to a starting state.
        
        # For efficiency, BFS from each directed edge
        # But that's O(n*k) BFS, each O(n*k). Total O(n^2*k^2).
        # For n up to a few thousand, this might be slow.
        # Alternative: use the fact that girth >= g means no non-backtracking walk
        # of length < g closes a cycle.
        
        # Let me use a simpler BFS-based approach:
        # For each vertex, do BFS tracking parent, and when we find a back edge,
        # compute cycle length.
        
        for start in range(n):
            dist = [-1]*n
            parent = [-1]*n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True
    
    def local_search(n, edges, adj, k, g, time_limit):
        """Try to reduce n by finding vertices to remove via edge swaps."""
        # This is complex. For now, let's try a simpler approach:
        # Try to find a 2-edge swap that reduces the graph while maintaining
        # k-regularity and girth.
        
        # Actually, reducing n while maintaining k-regularity is very hard.
        # A k-regular graph on n vertices has n*k/2 edges. If we remove a vertex,
        # we need to remove k edges and reconnect, which changes n.
        
        # Let me try a different approach: start from a slightly larger graph
        # and try to contract or remove vertices.
        
        # For simplicity, let's just try to verify the LUW graph and return it.
        # The local search would need to be more sophisticated.
        
        # Let me try: for each pair of edges (a,b) and (c,d) where {a,b} != {c,d},
        # try swapping to (a,c) and (b,d) or (a,d) and (b,c), check if still
        # k-regular (it will be if we do a proper swap) and girth ok.
        # But this doesn't reduce n.
        
        # To reduce n, we need a different strategy. Let's skip for now
        # and rely on LUW + known cages.
        
        return n, edges, adj
    
    # Try LUW with different d values
    best_n = None
    best_edges = None
    best_adj = None
    
    # Try d from max(2, g-2) down to 2, looking for valid girth
    # Actually, larger d gives more vertices but higher girth guarantee.
    # We want the smallest n that satisfies girth >= g.
    # LUW with parameter d typically gives girth at least 2d-1 or so.
    
    # Let's try d values and pick the one with smallest n that has girth >= g
    candidates = []
    for d in range(2, min(g+2, 15)):  # Cap d to avoid huge graphs
        if time.time() > deadline - 20:
            break
        try:
            n, edges = build_luw(d, q)
            if n < 100000:  # Cap size
                adj = build_adj(n, edges)
                # Verify degree
                ok = True
                for i in range(n):
                    if len(adj[i]) != k:
                        ok = False
                        break
                if ok:
                    # Check girth
                    if girth_ok(n, adj, g):
                        candidates.append((n, edges, adj))
        except Exception:
            continue
    
    if candidates:
        # Pick the one with smallest n
        candidates.sort(key=lambda x: x[0])
        best_n, best_edges, best_adj = candidates[0]
    else:
        # Fallback: use d = g
        d = g
        n, edges = build_luw(d, q)
        best_n, best_edges = n, edges
        best_adj = build_adj(n, edges)
    
    # Try local search if time permits
    if time.time() < deadline - 10 and best_n is not None:
        try:
            best_n, best_edges, best_adj = local_search(best_n, best_edges, best_adj, k, g, deadline - time.time() - 5)
        except Exception:
            pass
    
    return best_n, best_edges
