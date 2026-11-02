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

    def build_luw(d, rules=None):
        """Build LUW(d) graph over GF(q). Returns (n, edges)."""
        if rules is None:
            rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
            for c in range(5, d + 1):
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        else:
            rule = rules

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

    def check_girth_fast(n, edges, target_girth):
        """Fast girth check using adjacency lists. Returns True if girth >= target_girth."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            # BFS on directed edges (non-backtracking)
            # State: (prev, cur), distance from start
            from collections import deque
            visited = {}
            queue = deque()
            for s0 in adj[start]:
                visited[(start, s0)] = 1
                queue.append((start, s0))
            
            while queue:
                prev, cur = queue.popleft()
                length = visited[(prev, cur)]
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
                            visited[state] = length + 1
                            queue.append(state)
        return True

    def check_girth_nonbacktracking(n, edges, target_girth):
        """Check if girth >= target_girth using non-backtracking walks."""
        return check_girth_fast(n, edges, target_girth)

    target_girth = g if g % 2 == 0 else g + 1
    best_result = None
    
    # Search d=3 rules for q=5
    if q == 5:
        for a3 in [1, 2]:
            for b3 in [1, 2]:
                if time.time() > deadline: break
                rules = {2: (1, 1), 3: (a3, b3)}
                try:
                    n, edges = build_luw(3, rules)
                    if check_girth_nonbacktracking(n, edges, g):
                        if best_result is None or n < best_result[0]:
                            best_result = (n, edges)
                except Exception:
                    pass
        if best_result:
            return best_result

    # Search d=4 rules for q=5
    # Expand search: a3,b3 in {1,2}, a4,b4 in {1,2,3}
    for a3 in [1, 2]:
        for b3 in [1, 2]:
            for a4 in [1, 2, 3]:
                for b4 in [1, 2, 3]:
                    if time.time() > deadline: break
                    rules = {2: (1, 1), 3: (a3, b3), 4: (a4, b4)}
                    try:
                        n, edges = build_luw(4, rules)
                        if check_girth_nonbacktracking(n, edges, g):
                            if best_result is None or n < best_result[0]:
                                best_result = (n, edges)
                    except Exception:
                        pass
    
    if best_result:
        return best_result

    # Search d=5 rules for q=5 (if time permits)
    if q == 5 and time.time() < deadline - 30:
        # Try a few d=5 rule patterns
        # c=2:(1,1), c=3:(2,1), c=4:(1,2), c=5 varies
        for a5 in [1, 2, 3, 4]:
            for b5 in [1, 2, 3, 4]:
                if time.time() > deadline - 10: break
                rules = {2: (1, 1), 3: (2, 1), 4: (1, 2), 5: (a5, b5)}
                try:
                    n, edges = build_luw(5, rules)
                    if check_girth_nonbacktracking(n, edges, g):
                        if best_result is None or n < best_result[0]:
                            best_result = (n, edges)
                except Exception:
                    pass
        if best_result:
            return best_result

    # Fallback: standard LUW with sufficient d
    min_d = max(2, (target_girth - 2) // 2)
    for d in range(min_d, min_d + 6):
        if time.time() > deadline:
            break
        try:
            n, edges = build_luw(d)
            if check_girth_nonbacktracking(n, edges, g):
                return n, edges
        except Exception:
            continue
            
    d = max(4, (target_girth - 2) // 2 + 2)
    n, edges = build_luw(d)
    return n, edges
# EVOLVE-BLOCK-END
