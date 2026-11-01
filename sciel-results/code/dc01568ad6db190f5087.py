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
        """Check if girth >= target_girth using non-backtracking walks.
        For each vertex, do BFS on the non-backtracking graph (directed edges as states).
        A cycle of length L corresponds to a non-backtracking walk of length L that returns to start.
        We check if any non-backtracking closed walk of length < target_girth exists."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Non-backtracking BFS: state is (current_node, prev_node)
        # Start from each directed edge (start, neighbor)
        # We want to detect if we can return to start with length < target_girth
        
        for start in range(n):
            # dist[prev][cur] = shortest non-backtracking walk length from (start, prev) to cur
            # Actually, let's use: for each starting directed edge (start, s0), 
            # find shortest non-backtracking walk from s0 back to start
            visited = {}  # (prev, cur) -> length
            queue = []
            for s0 in adj[start]:
                visited[(start, s0)] = 1
                queue.append((start, s0, 1))
            
            qi = 0
            while qi < len(queue):
                prev, cur, length = queue[qi]
                qi += 1
                if length >= target_girth:
                    continue
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    if nxt == start:
                        # Found a cycle of length length + 1
                        if length + 1 < target_girth:
                            return False
                    else:
                        state = (cur, nxt)
                        if state not in visited:
                            visited[state] = length + 1
                            queue.append((cur, nxt, length + 1))
        return True

    # For LUW(d) over GF(q), the girth is at least 2d + 2
    # We need girth >= g
    # Since LUW is bipartite, girth is even, so we need girth >= g if g is even, girth >= g+1 if g is odd
    target_girth = g if g % 2 == 0 else g + 1
    
    # We need 2d + 2 >= target_girth, so d >= (target_girth - 2) / 2
    # But the theoretical bound may not be tight; try smaller d values too
    min_d = max(2, (target_girth - 2) // 2)
    
    # Try d values from a lower bound upward, checking girth explicitly
    best_result = None
    for d in range(2, min_d + 6):
        if time.time() > deadline:
            break
        try:
            n, edges = build_luw(d)
            if check_girth_nonbacktracking(n, edges, g):
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
                return n, edges
        except Exception:
            continue
    
    # Fallback: build with sufficient d
    d = max(4, (target_girth - 2) // 2 + 2)
    n, edges = build_luw(d)
    return n, edges
# EVOLVE-BLOCK-END
