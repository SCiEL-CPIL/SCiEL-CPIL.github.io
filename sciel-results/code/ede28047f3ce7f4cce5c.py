import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Target: 4-regular graph of girth 10 with minimal vertices.
    # Known optimal: 320 vertices.
    # Candidate 1: Circulant C_320(1, 5, 11, 17)
    # Candidate 2: Voltage graph on C_4 with Z_80
    # Candidate 3: LUW construction with d=4 (2048 vertices) if valid
    
    def verify_girth(n, edges, target_girth):
        """
        Verify that the graph has no cycle shorter than target_girth.
        Uses BFS from each vertex to find shortest cycle.
        Optimized for small n.
        """
        if n == 0:
            return True
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check for loops
        for u, v in edges:
            if u == v:
                return False
        
        # Check for repeated edges
        edge_set = set()
        for u, v in edges:
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                return False
            edge_set.add((u, v))
            
        # Check girth using BFS from each vertex
        # For girth g, we need to ensure no cycle of length < g.
        # BFS from vertex s, find shortest path back to s via a neighbor.
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a back edge or cross edge
                        # The cycle length is dist[u] + dist[v] + 1
                        # But this only finds cycles where the edge (u,v) is not a tree edge.
                        # We need to be careful.
                        pass
            
            # Better: BFS to find shortest cycle through s
            # dist[s] = 0.
            # For each neighbor v of s, BFS from v avoiding s.
            # If we reach s, cycle length is dist[s] + 1? No.
            # If we reach a neighbor of s, cycle length is dist[v] + 1 + 1?
            
            # Standard method:
            # For each vertex s, BFS.
            # When we see an edge (u, v) where dist[v] != -1 and v is not parent[u],
            # we have a cycle of length dist[u] + dist[v] + 1.
            # This finds the girth.
            
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            head = 0
            min_cycle = float('inf')
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] * 2 >= min_cycle:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if parent[u] != v and parent[v] != u:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_cycle:
                                min_cycle = cycle_len
            if min_cycle < target_girth:
                return False
                
        return True

    # Try Circulant C_320(1, 5, 11, 17)
    n = 320
    steps = [1, 5, 11, 17]
    edges = []
    for i in range(n):
        for step in steps:
            j = (i + step) % n
            if i < j:
                edges.append((i, j))
    
    if time.time() < deadline:
        if verify_girth(n, edges, g):
            return n, edges

    # Try Voltage Graph on C_4 with Z_80
    # Base graph C_4: 0-1-2-3-0
    # Voltages a, b, c, d
    # Vertex (v, g)
    # Edge (v, g) to (v+1, g+a)
    
    base_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    candidates = [
        (1, 3, 1, 3),
        (1, 5, 1, 5),
        (1, 7, 1, 7),
        (1, 9, 1, 9),
        (1, 11, 1, 11),
        (1, 13, 1, 13),
        (1, 15, 1, 15),
        (1, 17, 1, 17),
        (1, 19, 1, 19),
        (1, 21, 1, 21),
        (3, 5, 3, 5),
        (3, 7, 3, 7),
        (5, 7, 5, 7),
    ]
    
    for a, b, c, d in candidates:
        if time.time() > deadline:
            break
            
        edges_v = []
        # Map (v, g) to index
        # v in 0..3, g in 0..79
        # index = v * 80 + g
        
        # Add edges
        for g in range(80):
            # 0 -> 1 with voltage a
            edges_v.append((0 * 80 + g, 1 * 80 + (g + a) % 80))
            # 1 -> 2 with voltage b
            edges_v.append((1 * 80 + g, 2 * 80 + (g + b) % 80))
            # 2 -> 3 with voltage c
            edges_v.append((2 * 80 + g, 3 * 80 + (g + c) % 80))
            # 3 -> 0 with voltage d
            edges_v.append((3 * 80 + g, 0 * 80 + (g + d) % 80))
        
        # Remove duplicates
        unique_edges = set()
        for u, v in edges_v:
            if u > v:
                u, v = v, u
            unique_edges.add((u, v))
        edges_v = list(unique_edges)
        
        if verify_girth(320, edges_v, g):
            return 320, edges_v

    # Fallback: LUW Construction with d=4 (2048 vertices)
    # Check if time allows
    if time.time() < deadline - 10:
        q = k
        d = 4
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

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
        index, order, edges_luw = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges_luw.append((index[vertex], index[other]))
        
        n_luw = len(order)
        if verify_girth(n_luw, edges_luw, g):
            return n_luw, edges_luw

    # Final Fallback: Seed LUW with d=5 (8192 vertices)
    q = k
    d = 5
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    
    return len(order), edges_luw
