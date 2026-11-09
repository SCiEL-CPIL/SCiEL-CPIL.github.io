import time

def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    best_n = float('inf')
    best_edges = []
    
    # Determine starting d
    # Girth approx 2d+2.
    # 2d+2 >= g => d >= (g-2)/2
    start_d = max(2, (g - 2) // 2)
    
    for d in range(start_d, start_d + 5):
        if time.time() > deadline:
            break
            
        n, edges = _construct_luw(k, d)
        
        if n == 0:
            continue
            
        # Check if girth is at least g
        if _check_girth(n, edges, g):
            if n < best_n:
                best_n = n
                best_edges = edges
            # Since n grows with d, the first valid d is the best.
            break
            
    if best_n == float('inf'):
        # Fallback to a known valid large d if all small ones failed
        # This shouldn't happen for standard inputs
        n, edges = _construct_luw(k, start_d + 5)
        return n, edges
        
    return best_n, best_edges

def _construct_luw(k, d):
    q = k
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def _check_girth(n, edges, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    min_girth = float('inf')
    
    # BFS from each vertex
    for v in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[v] = 0
        queue = [v]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for w in adj[u]:
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    parent[w] = u
                    queue.append(w)
                else:
                    # If w is already visited and not the parent of u
                    # and u is not the parent of w
                    if w != parent[u] and u != parent[w]:
                        # Cycle found: v -> ... -> u -> w -> ... -> v
                        # Length is dist[u] + dist[w] + 1
                        # But this counts the cycle from v to u and v to w.
                        # The cycle is u-w + path(v,u) + path(v,w).
                        # Length = dist[u] + dist[w] + 1.
                        cycle_len = dist[u] + dist[w] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            
        if min_girth < g:
            return False
            
    return True
