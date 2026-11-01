"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import time
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin

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

    def check_girth_optimized(n, edges, target_girth):
        """Optimized girth check using non-backtracking BFS with array-based tracking."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For each start vertex, do non-backtracking BFS
        # State: (prev, cur) encoded as prev * n + cur for fast lookup
        # Use a list of booleans for visited states, reset per start vertex
        
        max_state = n * n
        visited = [False] * max_state
        
        for start in range(n):
            queue = []
            for s0 in adj[start]:
                state_id = start * n + s0
                visited[state_id] = True
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
                        if length + 1 < target_girth:
                            return False
                    else:
                        state_id = cur * n + nxt
                        if not visited[state_id]:
                            visited[state_id] = True
                            queue.append((cur, nxt, length + 1))
            
            # Reset visited for next start vertex - only reset what we set
            # Actually, we need to reset all visited states for this start
            # This is expensive, so use a different approach: timestamp-based
            # For now, just clear the list
            for s0 in adj[start]:
                visited[start * n + s0] = False
            # We need to clear all states that were set... this is complex
            # Let's use a set instead for correctness, but optimize differently
        
        return True

    # Better approach: use a set but with early termination and periodic time checks
    def check_girth_set(n, edges, target_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            visited = set()
            queue = []
            
            for s0 in adj[start]:
                state = (start, s0)
                visited.add(state)
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
                        if length + 1 < target_girth:
                            return False
                    else:
                        state = (cur, nxt)
                        if state not in visited:
                            visited.add(state)
                            queue.append((cur, nxt, length + 1))
            
            if (start + 1) % 5 == 0 and time.time() > deadline:
                return None
        
        return True

    # For LUW(d) over GF(q), girth >= 2d+2
    target_girth = g if g % 2 == 0 else g + 1
    min_d = max(2, (target_girth - 2) // 2)
    
    for d in range(min_d, min_d + 6):
        if time.time() > deadline:
            break
        try:
            n, edges = build_luw(d)
            result = check_girth_set(n, edges, g)
            if result is True:
                return n, edges
            elif result is None:
                if d == min_d:
                    return n, edges
        except Exception:
            continue
    
    d = max(4, (target_girth - 2) // 2 + 2)
    n, edges = build_luw(d)
    return n, edges
# EVOLVE-BLOCK-END
