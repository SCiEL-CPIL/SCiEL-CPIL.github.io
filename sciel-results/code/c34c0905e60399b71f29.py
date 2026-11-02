import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    q = k
    
    def build_luw(d):
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

    # Heuristic for minimal d
    # Girth of CD(d,q) is approx 2d+2
    # We want 2d+2 >= g => d >= (g-2)/2
    min_d = max(3, (g - 2) // 2)
    
    # Try d = min_d
    # If k=4, g=10, min_d = 4.
    # CD(4,4) has 512 vertices.
    # Seed uses d=5, 2048 vertices.
    
    if time.time() < deadline:
        n, edges = build_luw(min_d)
        # Optional: verify girth if n is small
        if n < 1000:
            # Quick girth check
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # BFS from each node to find shortest cycle
            # This is O(n * (k-1)^(g/2))
            # For n=512, g=10, depth=5, branching=3. 512 * 243 ~ 124k. Fast.
            girth_ok = True
            for i in range(n):
                visited = {i: 0}
                queue = deque([(i, 0)])
                while queue:
                    u, dist = queue.popleft()
                    if dist >= g // 2:
                        continue
                    for v in adj[u]:
                        if v not in visited:
                            visited[v] = dist + 1
                            queue.append((v, dist + 1))
                        else:
                            # If v is visited and v != u (back edge to parent is ok)
                            # Actually, if v is in visited and dist(v) < dist(u), 
                            # cycle length is dist(u) + dist(v) + 1?
                            # No, if we see a node already visited, it's a cycle.
                            # The length is dist(u) + 1 + dist(v) only if it's a cross edge?
                            # In BFS, if we see a visited node, the cycle length is dist(u) + dist(v) + 1?
                            # Not exactly. 
                            # Let's just check if any cycle is found < g.
                            # If dist(u) + 1 + dist(v) < g, then girth < g.
                            # But dist(v) is the distance from i.
                            # The cycle goes i -> ... -> v -> u -> ... -> i?
                            # No, the path is i -> ... -> u and i -> ... -> v.
                            # If u and v are connected, cycle is dist(u) + 1 + dist(v).
                            if dist(u) + 1 + dist(v) < g:
                                girth_ok = False
                                break
                if not girth_ok:
                    break
            
            if girth_ok:
                return n, edges
            else:
                # Try larger d
                pass

    # Fallback to original logic
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n, edges = build_luw(d)
    return n, edges
