import time
import collections

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    q = k
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    def build_luw(d):
        rule = {}
        if d >= 2: rule[2] = (1, 1)
        if d >= 3: rule[3] = (2, 1)
        if d >= 4: rule[4] = (1, 2)
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
            rule[c] = rules_cycle[r]

        index = {}
        order = []
        edges = []
        
        start = (0, (None,) + (0,) * d)
        index[start] = 0
        order.append(start)
        
        for i, vertex in enumerate(order):
            if time.time() > deadline - 10:
                return None
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                neighbor = (1 - side, tuple(y))
                
                if neighbor not in index:
                    index[neighbor] = len(order)
                    order.append(neighbor)
                
                if side == 0:
                    edges.append((index[vertex], index[neighbor]))
                    
        n = len(order)
        return n, edges

    def check_girth_fast(n, edges, min_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = collections.deque([start])
            
            while queue:
                curr = queue.popleft()
                if dist[curr] + 1 >= min_girth:
                    continue
                for neighbor in adj[curr]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[curr] + 1
                        parent[neighbor] = curr
                        queue.append(neighbor)
                    elif neighbor != parent[curr]:
                        cycle_len = dist[curr] + dist[neighbor] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    min_d = max(1, (g - 2) // 2)
    
    for d in range(min_d, 20):
        if time.time() > deadline - 20:
            break
            
        result = build_luw(d)
        if result is None:
            continue
            
        n, edges = result
        
        if check_girth_fast(n, edges, g):
            return n, edges
            
    return build_luw(min_d)
