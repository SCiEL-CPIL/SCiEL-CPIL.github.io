"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import time
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)

    def build_luw(d):
        """Build LUW(d) graph over GF(q). Returns (n, edges)."""
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
        return len(order), edges

    def check_girth_nonbacktracking(n, edges, target_girth):
        """Check girth >= target_girth using non-backtracking walks.
        
        A cycle of length L exists iff there is a non-backtracking closed walk of length L.
        We do BFS on the non-backtracking graph (directed edges as nodes) from each directed edge,
        looking for a path back to the reverse of the starting directed edge.
        
        For a k-regular graph, there are 2*n*k directed edges.
        We check for cycles of length 3, 4, ..., target_girth-1.
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Create directed edge index
        # directed edge (u, v) gets index u * k + pos where pos is position of v in adj[u]
        # Map (u, v) to a unique id
        dir_edge_id = {}
        dir_edges = []
        for u in range(n):
            for i, v in enumerate(adj[u]):
                eid = u * k + i
                dir_edge_id[(u, v)] = eid
                dir_edges.append((u, v))
        
        total_dir = 2 * n * k
        
        # Build non-backtracking adjacency: from directed edge (u,v), can go to (v,w) for w != u
        nb_adj = [[] for _ in range(total_dir)]
        for eid, (u, v) in enumerate(dir_edges):
            for w in adj[v]:
                if w != u:
                    nb_adj[eid].append(dir_edge_id[(v, w)])
        
        # For each directed edge, do BFS in the non-backtracking graph
        # looking for a path of length L-1 that reaches the reverse edge (v, u)
        # A cycle of length L starting with edge (u,v) ends when we reach directed edge (v,u)
        # after L-1 non-backtracking steps.
        
        # We need to check cycle lengths from 3 to target_girth - 1
        max_cycle_len = target_girth - 1
        
        # BFS from each directed edge, tracking distance
        # dist[eid] = shortest non-backtracking path length from start to eid
        # We want to find if any directed edge can reach its reverse in < target_girth steps
        
        # More efficient: for each start directed edge (u,v), BFS in nb graph
        # Check if we can reach (v,u) in L-1 steps for L < target_girth
        # i.e., in 2 to target_girth-2 steps (since L >= 3 means L-1 >= 2)
        
        max_steps = target_girth - 2  # L-1 where L < target_girth, so L-1 <= target_girth-2
        
        # Use BFS with early termination
        for start_eid in range(total_dir):
            u, v = dir_edges[start_eid]
            target_eid = dir_edge_id[(v, u)]
            
            if start_eid == target_eid:
                # This would be a 2-cycle, which can't happen in simple graph
                continue
            
            # BFS from start_eid
            dist = [-1] * total_dir
            dist[start_eid] = 0
            queue = [start_eid]
            qi = 0
            found_short_cycle = False
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                d = dist[node]
                if d >= max_steps:
                    continue
                for neighbor in nb_adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = d + 1
                        if neighbor == target_eid:
                            # Cycle length = d + 1 + 1 = d + 2
                            # Wait: we started at directed edge (u,v), took d+1 non-backtracking steps
                            # to reach (v,u). The cycle has length (d+1) + 1 = d + 2
                            # Actually: the non-backtracking walk has length d+1 (number of directed edges traversed)
                            # This corresponds to a cycle of length d+2
                            cycle_len = d + 1 + 1
                            if cycle_len < target_girth:
                                found_short_cycle = True
                                break
                        if d + 1 < max_steps:
                            queue.append(neighbor)
                    # If already visited, we don't need to revisit (BFS gives shortest)
                if found_short_cycle:
                    break
            if found_short_cycle:
                return False
        
        return True

    # For LUW(d) over GF(q), the girth is at least 2d + 2
    # We need girth >= g
    # Since LUW is bipartite, girth is even, so we need girth >= g if g is even, girth >= g+1 if g is odd
    target_girth = g if g % 2 == 0 else g + 1
    
    # We need 2d + 2 >= target_girth, so d >= (target_girth - 2) / 2
    min_d = max(2, (target_girth - 2) // 2)
    
    # Try d values starting from min_d
    best_result = None
    for d in range(min_d, min_d + 12):
        if time.time() > deadline:
            break
        try:
            n, edges = build_luw(d)
            # Verify girth using non-backtracking walks
            if check_girth_nonbacktracking(n, edges, target_girth):
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
                return n, edges
        except Exception:
            continue
    
    # Fallback: just build with a reasonable d
    d = max(4, (target_girth - 2) // 2 + 2)
    n, edges = build_luw(d)
    return n, edges
# EVOLVE-BLOCK-END
